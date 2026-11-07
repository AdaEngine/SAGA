import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const wasmPath = process.argv[2];
if (!wasmPath) throw new Error("Pass the built SAGAWebDiagnostic.wasm path");
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.SAGA_PLAYWRIGHT ?? "playwright-core");
const browser = await chromium.launch({
    executablePath: process.env.SAGA_CHROMIUM,
    args: ["--enable-unsafe-webgpu", ...(process.platform === "darwin" ? ["--use-angle=metal"] : [])],
});
try {
    const page = await browser.newPage();
    const root = "http://localhost:8799/";
    const mime = new Map([
        ["/", "text/html"],
        ["/reactor.mjs", "text/javascript"],
        ["/webgpu.mjs", "text/javascript"],
        ["/wasi.mjs", "text/javascript"],
        ["/SAGAWebDiagnostic.wasm", "application/wasm"],
    ]);
    await page.route(`${root}**`, async route => {
        const path = new URL(route.request().url()).pathname;
        const contentType = mime.get(path);
        if (!contentType) return route.fulfill({ status: 404 });
        const file = path.endsWith(".wasm") ? wasmPath : join(dirname(wasmPath), path === "/" ? "index.html" : path.slice(1));
        await route.fulfill({ contentType, body: await readFile(file) });
    });
    await page.goto(root);
    await page.waitForFunction(() => typeof globalThis.sagaDiagnostic?.then === "function");
    const result = await page.evaluate(() => globalThis.sagaDiagnostic);
    assert.equal(result.fallback, false);
    assert.equal(result.validation, "swift");
    assert.deepEqual(result.readback, [[0, 0, 0, 0], [8, 12, 16, 7]]);
    assert.equal(result.renderPixels, 56);
    assert.equal(result.vertexStreams, 2);
    assert.equal(result.indexedVertices, 6);
    assert.equal(result.instancesPerDraw, 2);
    assert.equal(result.renderProbe.length, 3);
    for (const [index, expected] of [242, 9, 4].entries()) {
        assert.ok(Math.abs(result.renderProbe[index] - expected) <= 1);
    }
    assert.equal(result.integerWords, 20);
    assert.equal(result.runtimeArrayWords, 5);
    assert.equal(result.storageTailWords, 5);
    assert.equal(result.storageVectors, 4);
    assert.deepEqual(result.groupedStorage, [29, 33, 23, 999]);
    assert.equal(result.builtinInvocations, 168);
    assert.deepEqual(result.workgroupWords, [18, 19, 20, 17, 118, 119, 120, 117, 218, 219, 220, 217]);
    assert.deepEqual(result.storageBarrierWords, [81, 80, 1081, 1080, 2081, 2080]);
    assert.equal(result.atomicTickets, 128);
    assert.equal(result.atomicRMWWords, 17);
    assert.equal(result.atomicCompareWords, 10);
    assert.equal(result.mixedAtomicWords, 17);
    assert.deepEqual(result.uniformLoadWords, [691, 691, 691, 691, 694, 694, 694, 694, 686, 686, 686, 686]);
    assert.deepEqual(result.selectWords, [3, 23, 40, 20, 60, 0xffffffff, 0xfffffffe, 1]);
    assert.deepEqual(result.constantWords, [40, 41, 42, 43]);
    assert.deepEqual(result.uniformWords, [30, 38, 17, 17]);
    assert.deepEqual(result.renderUniformPixels, [0xff0000ff, 0xff00ff00, 0xff0000ff, 0xff00ff00]);
    assert.deepEqual(result.matrixValues, [39, 54, 69, 69, 96, 123, 14, 19, 24, 20, 47, 24, 7, 8, 9, 0, 0, 0]);
    assert.deepEqual(result.matrixBufferValues, [0, 1, 55, 18]);
    assert.deepEqual(result.switchWords, [7, 7, 11, 19, 19, 215, 23, 1]);
    assert.deepEqual(result.textureUploadWords, [0xff445566, 0xff112233, 0xff445566, 0xff778899, 0xffaabbcc]);
    assert.deepEqual(result.textureLoadWords, [0, 0xffff0000, 0xff00ff00, 0, 0, 0, 0]);
    assert.deepEqual(result.renderTexturePixels, [0xffff0000, 0xff00ffff, 0xffff0000, 0xff0000ff]);
    assert.deepEqual(result.samplerValues, [0, 16, 0, 16, 10, 6, 0, 16, 16, 0, 0, 16, 0, 16, 0, 16]);
    assert.deepEqual(result.renderSamplerPixels, [0xffff0000, 0xffff60ff, 0xffff0000, 0xffff00ff]);
    assert.deepEqual(result.mipTextureWords, [0, 0xff00ff00, 0xffff0000, 0x3f000000, 0x3f000000]);
    assert.deepEqual(result.textureViewWords, [
        0x3f800000, 0, 0x40000000, 0x40000000, 0x3f800000, 0, 0x3f800000, 0x3f000000, 0xff00ff00,
    ]);
    assert.deepEqual(result.textureGatherLanes, [80, 100, 40, 20, 90, 110, 50, 30, 125, 85, 25, 65, 170, 180, 150, 140]);
    assert.deepEqual(result.storageTextureWords, [0, 0xffff00ff, 0xff0000ff]);
    assert.deepEqual(result.fragmentStorageTextureWords, [0xff0000ff, 0xffff0000, 0xff00ff00, 0xff00ff00, 0xff0000ff, 0xffff0000]);
    assert.deepEqual(result.readWriteStorageWords, [5, 0, 0, 10, 1, 20, 1, 0, 15, 20]);
    assert.deepEqual(result.textureBarrierWords, [81, 80, 1081, 1080, 2081, 2080]);
    assert.deepEqual(result.renderReadableStorageWords, [0xffff0000, 0xff00ffff, 15]);
    assert.deepEqual(result.depthAttachmentWords, [0xffff0000, 0xff00ff00, 0xffff0000, 0xff0000ff, 0x3e800000]);
    assert.deepEqual(result.depthTextureWords, [0xff0000ff, 0x3e800000, 0, 0x40000000, 0x3e800000]);
    assert.deepEqual(result.depthComparisonWords, [
        0xbf40ff00, 0xbf40ff00, 0, 0x3f800000, 0x3e800000, 0x3f400000,
        0x3e800000, 0x3f400000, 0x3f000000, 0x3f000000,
    ]);
    assert.deepEqual(result.depthGatherLanes, [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]);
    assert.deepEqual(result.rawDepthGather, [
        0.25, 0.5, 0.375, 0.125, 0.5, 0.875, 0.625, 0.375,
        0.875, 0.25, 0.125, 0.625, 0.25, 0.5, 0.375, 0.125,
    ]);
    const samples = result.multisampleWords.slice(0, 4);
    assert.ok(samples.includes(0) && samples.includes(1));
    assert.ok(samples.every(value => value === 0 || value === 1));
    assert.deepEqual(result.multisampleWords.slice(4, 8), samples.map(value => value ? 1 : 4));
    assert.deepEqual(result.multisampleWords.slice(8), [4, 4, 2, 0, 0xff000080, 0xff000080, 0, 0]);
    const signedPixel = [-17 >>> 0, 23, -3 >>> 0, 1];
    const unsignedPixel = [249, 1, 200, 7];
    assert.deepEqual(result.integerTextureWords, [
        ...Array.from({ length: 4 }, () => signedPixel).flat(), 0, 0, 0, 0,
        ...Array.from({ length: 4 }, () => unsignedPixel).flat(), 0, 0, 0, 0,
        ...unsignedPixel, 4, 4, 1, 1,
    ]);
    assert.deepEqual(result.integerGatherWords, [
        ...[-86, -76, -106, -116, -76, -66, -96, -106].map(value => value >>> 0),
        125, 85, 25, 65, 170, 180, 150, 140,
    ]);
    assert.deepEqual(result.sampleCoverageWords, [
        0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0, 1, 0, 0, 0, 0,
    ]);
    assert.deepEqual(result.fragmentDepthWords, [0xff0000ff, 0x3e800000, 0xffff0000, 0x3f800000]);
    const frontPixels = [0x01650101, 0x01650102, 0x01650201, 0x01650202];
    assert.deepEqual(result.fragmentRasterWords, [
        ...frontPixels, ...frontPixels.map(pixel => pixel & 0x00ffffff),
        0, 0, 0, 0, ...frontPixels, 0, 0, 0, 0,
    ]);
    assert.equal(result.interpolationWord, 0x543f2a15);
    const drawn = 0xff00ff00;
    const clear = 0xff000000;
    assert.deepEqual(result.topologyWords, [
        drawn, drawn, clear, drawn, clear, drawn, drawn, drawn, drawn,
        drawn, drawn, clear, drawn, clear, drawn, drawn, clear, drawn,
    ]);
    assert.deepEqual(result.blendWords, [0, 0x80ff4080, 0x804080bf]);
    assert.deepEqual(result.alphaCoverageWords.slice(0, 4), [0, 0, 0, 0]);
    const halfCoverage = result.alphaCoverageWords.slice(4, 8);
    assert.ok(halfCoverage.includes(0) && halfCoverage.includes(1));
    assert.deepEqual(result.alphaCoverageWords.slice(8), [0, 1, 0, 1, 0, 0, 0, 0, 0x3f400000]);
    assert.deepEqual(result.multipleColorWords, [0xff0000ff, 0x07210511, 0xffff8040]);
    assert.deepEqual(result.viewportWords, Array.from({ length: 16 }, (_, index) =>
        index === 9 || index === 10 ? [0xff00ff00, 0x3f000000] : [0, 0x3f400000]).flat());
    const biasRed = 0xff0000ff;
    assert.deepEqual(result.depthBiasWords, [
        biasRed, 0x3f000000, 0, 0x3f080000, biasRed, 0x3f100000,
        biasRed, 0x3ee00000, biasRed, 0x3ed00000, biasRed, 0x3eb00000,
    ]);
    const stencilPixels = [drawn, drawn, drawn, 0, 0, drawn, drawn, 0, 0, drawn];
    assert.deepEqual(result.stencilWords, [...stencilPixels, ...stencilPixels]);
    assert.deepEqual(result.readOnlyDepthWords, [drawn, drawn, 0, 0, 0x3e800000, 0x3e800000]);
    const sampledAspects = [1, 173, 2, 1, drawn, drawn, drawn, drawn];
    assert.deepEqual(result.aspectWords, [...sampledAspects, ...sampledAspects]);
    assert.deepEqual(result.stencilCopyWords, [0x00f08001, 0x00a50711, 1, 1, 3, 2]);
    assert.deepEqual(result.textureCopyWords, [0, 0xff0000ff]);
    assert.deepEqual(result.arrayWords, [1, 2, 128, 240, 204, 221, 238, 255, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 1, 0xf0800201, 0x04030201]);
    assert.deepEqual(result.layerViewWords, [0x011e140a, 0x01463c32, 0x281e140a, 0x50463c32, 0xff0000ff, 0, 0, 0]);
    assert.deepEqual(result.arraySamplingWords, Array.from(new Uint32Array(new Float32Array([
        1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0.25, 0.75, 1, 1, 1, 1, 0, 1, 0, 1,
    ]).buffer)));
    assert.deepEqual(result.storageArrayWords, [2, 1, 1, 2, 6, 10, 20, 0, 0, 0, 6, 0, 16, 0, 27, 0, 40, 0, 0, 0xffff00ff]);
    assert.equal(result.storageFormats, 16);
    assert.deepEqual(result.cubeWords, [...new Uint32Array(new Float32Array([
        0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1,
        1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 0.75, 1, 0, 1,
    ]).buffer), 0xbfffffff, 0xbfffffff]);
    assert.deepEqual(result.volumeWords, [...new Uint32Array(new Float32Array([
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 3, 1, 0, 0, 0, 0,
    ]).buffer), 0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00, 0, 0xff00ff00]);
    assert.equal(result.submissions, 70);
    assert.equal(result.remainingResources, 0);
    assert.equal(result.presentedFrames, 2);
    assert.deepEqual(result.surfaceSize, [8, 8]);
    const screenshot = await page.locator("#surface").screenshot();
    const presented = await page.evaluate(async png => {
        const image = await createImageBitmap(new Blob([new Uint8Array(png)], { type: "image/png" }));
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const pixels = canvas.getContext("2d");
        pixels.drawImage(image, 0, 0);
        const x = Math.floor(image.width / 16), y = Math.floor(image.height / 16);
        return {
            first: Array.from(pixels.getImageData(x, y, 1, 1).data),
            last: Array.from(pixels.getImageData(image.width - x - 1, image.height - y - 1, 1, 1).data),
        };
    }, [...screenshot]);
    for (const [index, expected] of [242, 9, 4, 255].entries()) {
        assert.ok(Math.abs(presented.first[index] - expected) <= 1, JSON.stringify(presented));
    }
    assert.deepEqual(presented.last, [0, 0, 255, 255]);
    const lifecycle = await page.evaluate(async () => {
        const { runDiagnostic, closeDiagnosticCanvas } = await import("/reactor.mjs");
        const surface = document.querySelector("#surface");
        const failure = promise => promise.then(() => "unexpected success", error => error.message);
        const duplicate = await failure(runDiagnostic(undefined, surface));
        closeDiagnosticCanvas(surface);
        const opening = runDiagnostic(undefined, surface);
        closeDiagnosticCanvas(surface);
        const cancelled = await failure(opening);
        const retry = await runDiagnostic(undefined, surface);
        closeDiagnosticCanvas(surface);
        const offscreen = await runDiagnostic();
        return { duplicate, cancelled, retryFrames: retry.presentedFrames, offscreenFrames: offscreen.presentedFrames };
    });
    assert.match(lifecycle.duplicate, /Canvas already belongs to a diagnostic/);
    assert.match(lifecycle.cancelled, /Canvas diagnostic closed/);
    assert.equal(lifecycle.retryFrames, 2);
    assert.equal(lifecycle.offscreenFrames, 0);
    await page.route(`${root}SAGAWebDiagnostic.wasm`, route => route.fulfill({ status: 404 }));
    await page.reload();
    await page.waitForFunction(() => typeof globalThis.sagaDiagnostic?.then === "function");
    const withoutWasm = await page.evaluate(() => globalThis.sagaDiagnostic.then(
        () => "unexpected success", error => error.message,
    ));
    assert.match(withoutWasm, /Swift\/Wasm download failed/);
    await page.unroute(`${root}SAGAWebDiagnostic.wasm`);
    await page.addInitScript(() => Object.defineProperty(navigator, "gpu", { value: undefined }));
    await page.reload();
    await page.waitForFunction(() => typeof globalThis.sagaDiagnostic?.then === "function");
    const withoutGPU = await page.evaluate(() => globalThis.sagaDiagnostic.then(
        () => "unexpected success", error => error.message,
    ));
    assert.match(withoutGPU, /No hardware WebGPU adapter available/);
    console.log(JSON.stringify(result));
} finally {
    await browser.close();
}
