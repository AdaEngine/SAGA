import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { basename } from "node:path";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.SAGA_PLAYWRIGHT ?? "playwright-core");
const browser = await chromium.launch({
    executablePath: process.env.SAGA_CHROMIUM,
    args: ["--enable-unsafe-webgpu", "--use-angle=metal"],
});
try {
    const page = await browser.newPage();
    const root = "http://localhost:8799/";
    const modules = new Map(await Promise.all(["webgpu.mjs", "golden.mjs"].map(async file =>
        [file, await readFile(new URL(`../web/${file}`, import.meta.url))])));
    await page.route(root + "**", route => {
        const module = modules.get(basename(new URL(route.request().url()).pathname));
        return module ? route.fulfill({ contentType: "text/javascript", body: module })
            : route.fulfill({ contentType: "text/html", body: "<canvas></canvas>" });
    });
    await page.goto(root);
    const result = await page.evaluate(async () => {
        const { openWebGPU } = await import("./webgpu.mjs");
        const host = await openWebGPU(null, { deferCanvas: true });
        try {
            host.attachCanvas(document.querySelector("canvas"));
            function packet(operation, ...words) {
                const out = new Uint8Array(20 + 4 * words.length);
                const view = new DataView(out.buffer);
                view.setUint32(0, operation, true);
                view.setBigUint64(4, 1n, true);
                view.setBigUint64(12, 2n, true);
                words.forEach((word, i) => view.setUint32(20 + 4 * i, word, true));
                return out;
            }
            await host.handle(packet(8, 2, 2, 2, GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST, 1, 1, 1, 0));
            // Submit one writeTexture command. The texture and wire data are BGRA.
            const writeHeader = [6, 1, 17, 1, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 8, 0, 16];
            const upload = new Uint8Array(writeHeader.length * 4 + 16);
            const uploadView = new DataView(upload.buffer);
            writeHeader.forEach((word, i) => uploadView.setUint32(i * 4, word, true));
            for (let i = writeHeader.length * 4; i < upload.length; i += 4) upload.set([7, 13, 211, 255], i);
            await host.handle(upload);
            const capture = host.captureNextFrame({ signal: AbortSignal.timeout(5000) });
            await host.handle(packet(12));
            const frame = await capture;
            const { compareFrames, frameFromPng, frameToPng } = await import("./golden.mjs");
            const roundtrip = await frameFromPng(await frameToPng(frame));
            const dark = { ...roundtrip, pixels: roundtrip.pixels.map((byte, i) => i % 4 === 3 ? byte : byte / 2) };
            return { width: frame.width, height: frame.height, pixels: [...frame.pixels],
                match: compareFrames(frame, roundtrip).ok, rejectsDarkened: !compareFrames(frame, dark).ok, ...host.metrics() };
        } finally { host.close(); }
    });
    assert.equal(result.presentedFrames, 1);
    assert.deepEqual(result.surfaceSize, [2, 2]);
    assert.equal(result.width, 2);
    assert.equal(result.height, 2);
    assert.deepEqual(result.pixels, Array(4).fill([211, 13, 7, 255]).flat());
    assert.equal(result.match, true);
    assert.equal(result.rejectsDarkened, true);
    console.log("Deferred canvas presented one frame; WebGPU readback returned all 16 expected bytes.");
} finally { await browser.close(); }
