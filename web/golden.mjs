// Compare presentation readbacks without reading the WebGPU canvas itself.

function canvasFor(frame) {
    const canvas = new OffscreenCanvas(frame.width, frame.height);
    canvas.getContext("2d").putImageData(new ImageData(frame.pixels, frame.width, frame.height), 0, 0);
    return canvas;
}

function resize(frame, width, height) {
    if (frame.width === width && frame.height === height) return frame;
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(canvasFor(frame), 0, 0, width, height);
    return { width, height, pixels: context.getImageData(0, 0, width, height).data };
}

export function compareFrames(actual, expected, { channelTolerance = 12, maxBadFraction = 0.02, maxMeanError = 4 } = {}) {
    if (actual.width !== expected.width || actual.height !== expected.height) throw new Error("Golden frame size differs");
    const count = actual.width * actual.height;
    if (!count || actual.pixels.length !== count * 4 || expected.pixels.length !== count * 4) {
        throw new Error("Golden frame has invalid pixel data");
    }
    let sum = 0;
    let bad = 0;
    for (let i = 0; i < count * 4; i += 4) {
        let worst = 0;
        for (let channel = 0; channel < 4; channel++) {
            const delta = Math.abs(actual.pixels[i + channel] - expected.pixels[i + channel]);
            sum += delta;
            worst = Math.max(worst, delta);
        }
        if (worst > channelTolerance) bad++;
    }
    const meanError = sum / (count * 4);
    const badFraction = bad / count;
    return { ok: meanError <= maxMeanError && badFraction <= maxBadFraction, meanError, badFraction };
}

export async function captureStable(host, { width = 160, height = 96 } = {}) {
    let previous;
    for (let attempt = 0; attempt < 8; attempt++) {
        const frame = resize(await host.captureNextFrame({ signal: AbortSignal.timeout(10_000) }), width, height);
        if (previous && compareFrames(frame, previous, { channelTolerance: 2, maxBadFraction: 0, maxMeanError: 0.5 }).ok) {
            return frame;
        }
        previous = frame;
    }
    throw new Error("Browser frames did not settle");
}

export function frameToPng(frame) {
    return canvasFor(frame).convertToBlob({ type: "image/png" });
}

export async function frameFromPng(blob) {
    const bitmap = await createImageBitmap(blob);
    try {
        const { width, height } = bitmap;
        const canvas = new OffscreenCanvas(width, height);
        const context = canvas.getContext("2d", { willReadFrequently: true });
        context.drawImage(bitmap, 0, 0);
        return { width, height, pixels: context.getImageData(0, 0, width, height).data };
    } finally {
        bitmap.close();
    }
}

export async function checkGolden(host, url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load golden frame (${response.status})`);
    const expected = await frameFromPng(await response.blob());
    const actual = await captureStable(host, { width: expected.width, height: expected.height });
    return { ...compareFrames(actual, expected), actual };
}
