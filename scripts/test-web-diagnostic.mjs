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
    assert.equal(result.runtimeArrayWords, 17);
    assert.equal(result.storageTailWords, 5);
    assert.equal(result.storageVectors, 4);
    assert.deepEqual(result.groupedStorage, [29, 33, 23, 999, 119, 0xffffffec, 49, 999]);
    assert.equal(result.builtinInvocations, 342);
    const adapterLimits = await page.evaluate(async () => {
        const limits = (await navigator.gpu.requestAdapter()).limits;
        return {
            storage: limits.maxComputeWorkgroupStorageSize, invocations: limits.maxComputeInvocationsPerWorkgroup,
            size: { x: limits.maxComputeWorkgroupSizeX, y: limits.maxComputeWorkgroupSizeY, z: limits.maxComputeWorkgroupSizeZ },
            dispatch: limits.maxComputeWorkgroupsPerDimension,
            vertexAttributes: limits.maxVertexAttributes,
            vertexBuffers: limits.maxVertexBuffers,
            vertexStride: limits.maxVertexBufferArrayStride,
            interstage: limits.maxInterStageShaderVariables,
            colorAttachments: limits.maxColorAttachments,
            colorBytes: limits.maxColorAttachmentBytesPerSample,
        };
    });
    assert.equal(result.maxComputeWorkgroupStorageSize, adapterLimits.storage);
    assert.equal(result.maxComputeInvocationsPerWorkgroup, adapterLimits.invocations);
    assert.deepEqual(result.maxComputeWorkgroupSize, adapterLimits.size);
    assert.equal(result.maxComputeWorkgroupsPerDimension, adapterLimits.dispatch);
    assert.equal(result.maxVertexAttributes, adapterLimits.vertexAttributes);
    assert.equal(result.maxVertexBuffers, adapterLimits.vertexBuffers);
    assert.equal(result.maxVertexBufferArrayStride, adapterLimits.vertexStride);
    assert.equal(result.maxInterStageShaderVariables, adapterLimits.interstage);
    assert.equal(result.maxColorAttachments, adapterLimits.colorAttachments);
    assert.equal(result.maxColorAttachmentBytesPerSample, adapterLimits.colorBytes);
    assert.deepEqual(result.workgroupWords, [18, 19, 20, 17, 118, 119, 120, 117, 218, 219, 220, 217, 81, 82, 83, 80,
        Math.min(512, adapterLimits.size.x, adapterLimits.invocations),
        329, 330, 331, 328, 429, 430, 431, 428, 529, 530, 531, 528]);
    assert.deepEqual(result.storageBarrierWords, [
        81, 80, 1081, 1080, 2081, 2080, 74, 73, 202, 201, 330, 329, 84, 83, 212, 211, 340, 339,
    ]);
    assert.equal(result.atomicTickets, 384);
    assert.equal(result.atomicRMWWords, 44);
    assert.equal(result.atomicCompareWords, 24);
    assert.equal(result.mixedAtomicWords, 17);
    assert.deepEqual(result.uniformLoadWords, [
        691, 691, 691, 691, 694, 694, 694, 694, 686, 686, 686, 686,
        1101, 1102, 1103, 1104, 1302, 1303, 1304, 1301, 1501, 1502, 1503, 1504,
    ]);
    assert.deepEqual(result.selectWords, [3, 23, 40, 20, 60, 0xffffffff, 0xfffffffe, 1]);
    assert.deepEqual(result.constantWords, [40, 41, 42, 43]);
    assert.deepEqual(result.uniformWords, [30, 38, 17, 17]);
    const packedVertexFormats = result.supportsVertexFormatSnorm1010102 ? 27 : 26;
    const packedBatchSize = Math.min(15, result.maxVertexAttributes - 1);
    const packedVertexMasks = [];
    for (let index = 0; index < packedVertexFormats; index += packedBatchSize) {
        packedVertexMasks.push(2 ** Math.min(packedBatchSize, packedVertexFormats - index) - 1);
    }
    assert.deepEqual(result.renderUniformPixels,
        [0xff0000ff, 0xff00ff00, 0xff0000ff, 0xff00ff00, 0xff00004d, ...packedVertexMasks]);
    assert.deepEqual(result.matrixValues, [
        39, 54, 69, 69, 96, 123, 14, 19, 24, 20, 47, 24, 7, 8, 9, 0, 0, 0,
        14, 11, -12, 33, 25, -27, 63, 43, -45, 93, 61, -63, 1, 4, -2, 5, 3, -6,
    ]);
    assert.deepEqual(result.matrixBufferValues, [0, 1, 55, 18, 2, 21, 45, 15]);
    assert.deepEqual(result.switchWords, [7, 7, 11, 19, 39, 219, 23, 1, 28, 28, 50, 99, 38, 22, 64, 99, 40, 43, 0, 10, 20, 99, 40, 3]);
    assert.deepEqual(result.textureUploadWords, [0xff445566, 0xff112233, 0xff445566, 0xff778899, 0xffaabbcc]);
    assert.deepEqual(result.textureLoadWords, [0, 0xffff0000, 0xff00ff00, 0, 0, 0, 0]);
    assert.deepEqual(result.renderTexturePixels, [0xffff0000, 0xff00ffff, 0xffff0000, 0xff0000ff]);
    assert.deepEqual(result.samplerValues, [0, 16, 0, 16, 10, 6, 0, 16, 16, 0, 0, 16, 0, 16, 0, 16]);
    assert.deepEqual(result.renderSamplerPixels, [0xffff0000, 0xffff60ff, 0xffff0000, 0xffff00ff]);
    assert.deepEqual(result.mipTextureWords, [
        0, 0xff00ff00, 0xffff0000, 0x3f000000, 0x3f000000,
        0, 0x3f800000, 0, 0x3f800000, 0, 0, 0x3f800000, 0x3f800000,
        0, 0x3f000000, 0x3f000000, 0x3f800000, 0x3f800000, 0, 0, 0x3f800000,
        0x40a00000, 0x40400000, 0x40000000, 0x40400000,
    ]);
    assert.deepEqual(result.textureViewWords, [
        0x3f800000, 0, 0x40000000, 0x40000000, 0x3f800000, 0, 0x3f800000, 0x3f000000, 0xff00ff00,
    ]);
    assert.deepEqual(result.textureGatherLanes, [
        80, 100, 40, 20, 90, 110, 50, 30, 125, 85, 25, 65, 170, 180, 150, 140,
        90, 110, 50, 30,
    ]);
    assert.deepEqual(result.storageTextureWords, [0, 0xffff00ff, 0xff0000ff]);
    assert.deepEqual(result.fragmentStorageTextureWords, [0xff0000ff, 0xffff0000, 0xff00ff00, 0xff00ff00, 0xff0000ff, 0xffff0000]);
    assert.deepEqual(result.readWriteStorageWords, [5, 0, 0, 10, 1, 20, 1, 0, 15, 20, 20, 0, 0, 1, 2, 1, 0x01020318, 20]);
    assert.deepEqual(result.textureBarrierWords, [81, 80, 1081, 1080, 2081, 2080]);
    assert.deepEqual(result.renderReadableStorageWords, [0xffff0000, 0xff00ffff, 15]);
    assert.deepEqual(result.depthAttachmentWords, [0xffff0000, 0xff00ff00, 0xffff0000, 0xff0000ff, 0x3e800000]);
    assert.deepEqual(result.depthTextureWords, [0xff0000ff, 0x3e800000, 0, 0x40000000, 0x3e800000]);
    assert.deepEqual(result.depthComparisonWords, [
        0xbf40ff00, 0xbf40ff00, 0, 0x3f800000, 0x3e800000, 0x3f400000,
        0x3e800000, 0x3f400000, 0x3f000000, 0x3f000000,
        0x3f800000, 0,
    ]);
    assert.deepEqual(result.depthGatherLanes, [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]);
    assert.deepEqual(result.rawDepthGather, [
        0.25, 0.5, 0.375, 0.125, 0.5, 0.875, 0.625, 0.375,
        0.875, 0.25, 0.125, 0.625, 0.25, 0.5, 0.375, 0.125,
    ]);
    const samples = result.multisampleWords.slice(0, 4);
    assert.ok(samples.includes(0) && samples.includes(1));
    assert.ok(samples.every(value => value === 0 || value === 1));
    assert.deepEqual(result.multisampleWords.slice(4, 8), samples.map(value => value ? 1 : 4));
    assert.deepEqual(result.multisampleWords.slice(8), [4, 4, 2, 0, 0xff000080, 0xff000080, 0, 0, 4, 1, 4, 1,
        0xff0000ff, 0]);
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
    const sampleOneMarker = result.sampleCoverageWords.slice(5, 7);
    const sampleThreeMarker = result.sampleCoverageWords.slice(13, 15);
    assert.notDeepEqual(sampleOneMarker, sampleThreeMarker);
    assert.ok([...sampleOneMarker, ...sampleThreeMarker].every(coordinate => coordinate >= 0 && coordinate <= 128));
    assert.deepEqual(result.sampleCoverageWords, [
        0, 0, 0, 0, 2, ...sampleOneMarker, 1, 0, 0, 0, 0, 4, ...sampleThreeMarker, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0, 1, 0, 0, 0, 0,
    ]);
    assert.deepEqual(result.fragmentDepthWords, [0xff0000ff, 0x3e800000, 0xffff0000, 0x3f800000, 0x3f800000, 0xff00ff00,
        0x3f400000, 0x3e000000]);
    const frontPixels = [0x01650101, 0x01650102, 0x01650201, 0x01650202];
    assert.deepEqual(result.fragmentRasterWords, [
        ...frontPixels, ...frontPixels.map(pixel => pixel & 0x00ffffff),
        0, 0, 0, 0, ...frontPixels, 0, 0, 0, 0, 0xff004039, 0,
        0xffffff00, 0xffffffff, 0xffff0000, 0xffff00ff,
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
    const wideColorCount = Math.min(result.maxColorAttachments, Math.floor(result.maxColorAttachmentBytesPerSample / 16));
    const wideColors = Array.from({ length: wideColorCount }, (_, index) => [index + 0.25, 0.5, 0.75, 1]).flat();
    assert.deepEqual(result.multipleColorWords,
        [0xff0000ff, 0x07210511, 0xffff8040, ...new Uint32Array(new Float32Array(wideColors).buffer), 0xff00ff00, 1, 0, 2]);
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
    assert.deepEqual(result.stencilCopyWords, [0x00f08001, 0x00a50711, 1, 1, 3, 2, 1, 0, 3, 2, 0, 17, 3, 2]);
    assert.deepEqual(result.textureCopyWords, [0, 0xff0000ff]);
    assert.deepEqual(result.arrayWords, [1, 2, 128, 240, 204, 221, 238, 255, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 1,
        0xf0800201, 0x04030201, 0xf0800201, 0x04030201, 0x08070605, 0xffeeddcc]);
    assert.deepEqual(result.layerViewWords, [0x011e140a, 0x01463c32, 0x281e140a, 0x50463c32, 0xff0000ff, 0, 0, 0]);
    assert.deepEqual(result.arraySamplingWords, Array.from(new Uint32Array(new Float32Array([
        1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0.25, 0.75, 1, 1, 1, 1, 0, 1, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    ]).buffer)));
    assert.deepEqual(result.storageArrayWords, [
        2, 1, 1, 2, 6, 10, 20, 0, 0, 0, 6, 0, 16, 0, 27, 0, 40, 0, 0, 0xffff00ff,
        16, 0, 0, 0, 0, 0, 0, 27, 1, 1, 6, 0, 61, 0, 72, 0, 40, 0,
    ]);
    assert.equal(result.storageFormats, 16);
    assert.deepEqual(result.cubeWords, [...new Uint32Array(new Float32Array([
        0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1,
        1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 0.75, 1, 0, 1,
    ]).buffer), 0xbfffffff, 0xbfffffff]);
    assert.deepEqual(result.volumeWords, [...new Uint32Array(new Float32Array([
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 3, 1, 0, 0, 0, 0, 1, 0, 0, 1,
    ]).buffer), 0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00, 0, 0xff00ff00]);
    assert.deepEqual(result.volumeStorageWords, [2, 1, 2, 7, 20, 0, 0, 1, ...Array(12).fill(0), 0, 8, 27, 21, 0, 0xffff00ff]);
    assert.deepEqual(result.volumeRenderWords, [
        0xff, 0xff, 0xff00ff40, 0xff00ff40, 0xff00ffff, 0xff00ffff,
        0xffff, 0xff, 0xff00ffff, 0xff00ff40, 0xff00ffff, 0xff00ffff,
    ]);
    assert.deepEqual(result.lineTextureWords, [...new Uint32Array(new Float32Array([
        1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 2, 2, 0, 0, 0, 0,
    ]).buffer), 0, 2, 0xff00ff00]);
    const inputBits = [0x3f800000, 0xc0200000, 0x3f000000, 0x00800000];
    assert.deepEqual(result.bitcastWords, [...inputBits, ...inputBits, ...inputBits,
        0x3f800000, 0xc0200000, 0x00800000, 0x40000000, 0x3f800000, 0xc0200000, 0x80000000, 0xffffffff]);
    const eagerBooleanWords = [
        0, 1, 1234, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 12345678, 1, 0, 1,
    ];
    assert.deepEqual(result.integerBitWords, [
        32, 31, 0, 0, 32, 0, 31, 0, 0, 1, 1, 32, 0, 0x80000000, 1, 0xffffffff,
        0xffffffff, 0, 31, 31, 0xffffffff, 0, 30, 0xffffffff, 0xffffffff, 0, 31, 0,
        0, 0, 8, 15, 0, 0, 0xfffffff8, 0xffffffff, 0x50000000, 0x50000001, 0x50000000, 0x5fffffff,
        0, 1, 0x80000000, 0xffffffff, 0, 1, 0x80000000, 0xffffffff, 0, 0xffffffff, 0, 7,
        32, 32, 32, 0x80000000,
        ...eagerBooleanWords,
        202, 900, 11, 0, 101, 202, 33, 900,
        0x80000000, 0x01234561, 0xfffffff2, 0x12345673, 0xfffffff8, 0xffffffff, 0xffffffff, 7, 8, 15, 15, 7,
        0x10000001, 0xf6a2c480, 0xffffffff, 0xee6a2c48, 2, 13, 32, 15,
        0x80000008, 0x0123456f, 0xffffffff, 0x12345677, 0, 0, 0, 0, 0, 0, 0, 0,
        0x10000001, 0xf6a2c480, 0xffffffff, 0xee6a2c48, 2, 13, 32, 15,
        0x40000004, 0x0f123456, 1, 0x12345677, 0xc0000004, 0xff123456, 0xffffffff, 0x12345677,
        0x10, 0x123456f0, 0x80000000, 0x12345677,
        8, 0, 0xffffffff, 0x12345677, 0xfffffff8, 0, 0xffffffff, 0x12345677,
        0x80000000, 0x80000000, 0xffffffff, 0x12345677, 0x80000000, 0x80000000, 0xffffffff, 0x12345677,
        8, 0, 15, 1, 0xfffffff8, 0, 0xffffffff, 1,
        0x50000008, 0x5123456f, 0x5fffffff, 0x52345677, 0xe0000008, 0xe123456f, 0xefffffff, 0xe2345677,
        0, 0, 0, 0, 0x80000008, 0x0123456f, 0xffffffff, 0x12345677, 0x80000008, 0x0123456f, 0xffffffff, 0x12345677,
        0x10000001, 0xf6a2c480, 0xffffffff, 0xee6a2c48, 2, 13, 32, 15,
        ...eagerBooleanWords,
    ]);
    assert.deepEqual(result.packingWords, [
        0x7f40c181, 0xff800000, 0x4000c001, 0xffff8000,
        0x807fff7f, 0x7f7fff80, 0x78ffff00, 0xffffffff,
        0xffffff80, 0xffffffff, 0, 127, 128, 255, 0, 127,
        0xbf800000, 0xbf800000, 0, 0x3f800000, 0, 0x3f800000, 0, 0x3f800000,
        0xbf800000, 0x3f800000, 0, 0x3f800000,
        0xc0003c00, 0x3f800000, 0xc0000000, 0xc0003c00,
        0x7f40c181, 0xff800000, 0x4000c001, 0xc0003c00,
    ]);
    assert.deepEqual(result.geometricWords, [...new Uint32Array(new Float32Array([
        25, 5, 5, 5, 4, -3, 0, 4, 0, 1, 0, 0,
        0, 1, 0, -1, 0, 1, 0, -1, 0, 0, 0, 0,
    ]).buffer), 3, 0, 0xfffffe7c, 1404]);
    assert.deepEqual(result.matrixBuiltinWords.slice(0, 18), [...new Uint32Array(new Float32Array([
        -3, 1, 24, -2, 1, 2, 3, 6, 14, 32, 4, 5, 1, 4, 2, 5, 3, 6,
    ]).buffer)]);
    const inverseExpected = [
        0.5, 0, -0.125, 0.25, -24, 20, -5, 18, -15, 4, 5, -4, 1,
        0.5, 0, 0, 0, -0.125, 0.25, 0, 0, 0.03125, -0.0625, 0.125, 0, -0.0078125, 0.015625, -0.03125, 0.0625,
    ];
    const inverseEnd = 18 + inverseExpected.length;
    const inverses = new Float32Array(new Uint32Array(result.matrixBuiltinWords.slice(18, inverseEnd)).buffer);
    assert.equal(inverses.length, inverseExpected.length);
    for (const [index, expected] of inverseExpected.entries()) {
        assert.ok(Math.abs(inverses[index] * 1e-12 - expected) <= Math.max(1, Math.abs(expected)) * 0.0008);
    }
    const matrixInstructions = new Float32Array(new Uint32Array(result.matrixBuiltinWords.slice(inverseEnd)).buffer);
    assert.deepEqual(Array.from(matrixInstructions), [2, 5, 16, 12, 62, 152, 39, 54, 69, 36, 88, 39]);
    assert.deepEqual(result.derivativeWords, Array(12).fill(0x09050502));
    assert.deepEqual(result.derivativeFilterWords, Array(4).fill(0xff00ff00));
    assert.deepEqual(result.numericBuiltinWords.slice(0, 52), [
        0x80000000, 7, 0, 9, 0xffffffff, 0xffffffff, 0, 1,
        3, 1, 3, 2, 0xffffffff, 3, 8, 3, 4, 2, 3, 1,
        ...new Uint32Array(new Float32Array([
            -2, -2, 2, 2, -3, -2, 1, 2, -2, -1, 2, 3, -2, -1, 1, 2,
            0.5, 0.5, 0.5, 0.5, 0, 0, 1, 1, 2, 0.5, 0, 1, 2.5, -1, -2.5, 2.5,
        ]).buffer),
    ]);
    const spirvMath = result.numericBuiltinWords.slice(52, 60);
    assert.equal(spirvMath.length, 8);
    const [degrees, hyperbolic] = new Float32Array(new Uint32Array(spirvMath.slice(0, 2)).buffer);
    assert.ok(Math.abs(degrees - 90) < 0.001);
    assert.ok(Math.abs(hyperbolic - 0.54930615) < 0.00001);
    assert.deepEqual(spirvMath.slice(2), [
        5, 0xff00ff00, ...new Uint32Array(new Float32Array([5, 2, -3]).buffer), 2,
    ]);
    assert.deepEqual(result.numericBuiltinWords.slice(60, 120), [
        1, 0, 0, 0, 1, 0, 0, 1,
        0, 1, 1, 0, 0, 1, 1, 1,
        0, 1, 0, 0, 0, 1, 0, 1,
        0, 0, 1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 1, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 1,
        0, 0, 0, 1, 0, 1, 1, 0,
        1, 0, 1, 0,
    ]);
    const remainderBits = new Uint32Array(new Float32Array([-2.5, 2.5, -2.5, 2.5, 0.5, -0.5, -2.5, 2.5]).buffer);
    assert.deepEqual(result.numericBuiltinWords.slice(120), [
        2, 0xffffffff, 1, 0xfffffffe, ...remainderBits, 2, remainderBits[0], remainderBits[4], 0,
    ]);
    const expectedTranscendental = [
        0.47942555, 0.87758255, 0.5463025, 2.3561945,
        0.5235988, 1.0471976, 0.7853982, 135,
        0.5210953, 1.127626, -0.46211717, 3.1415927,
        -0.48121184, 1.316958, 0.54930615, 3,
        1.6487213, 4, 0.6931472, 3, 2, 4, 8, 16,
    ];
    const transcendental = new Float32Array(new Uint32Array(result.transcendentalWords).buffer);
    assert.equal(transcendental.length, expectedTranscendental.length);
    for (const [index, expected] of expectedTranscendental.entries()) {
        assert.ok(Math.abs(transcendental[index] - expected) <= Math.max(1, Math.abs(expected)) * 0.0004);
    }
    assert.deepEqual(result.decompositionWords, [...new Uint32Array(new Float32Array([
        0.5, -0.5, -0.8125, 0.6875, 4, -2, 2, 2,
        -0.25, 0.75, -3, 2, -0.5, -2, -0.25, 0.25,
        3.4028234663852886e38, -3.25, 0.125, 2 ** -126,
        -0.203125, 1, 2 ** 127, 2 ** 127,
        -0.25, 0.125, 0.5, -3, 0, 8, -0.8125, 0.5, 0.53125,
    ]).buffer), 2, 0xfffffffe, 4]);
    if (result.halfPrecisionWords !== null) {
        assert.deepEqual(result.halfPrecisionWords, [...new Uint32Array(new Float32Array([
            65504, -3.25, 2.75, 2 ** -14, -0.25, 0.75, 3, -2, 0.99951171875, 2, -0.8125, 2,
        ]).buffer)]);
    }
    if (result.halfBufferWords !== null) {
        assert.deepEqual(result.halfBufferWords, [
            0xc7004880, 0x44803e00, 0x38004780, 0x38003800,
            0x40003c00, 0xc7004100, 0x45004400, 0xc7004600, 0x48004700, 0xc7004880,
            0xc7004900, 0x49804300, 0xbf004ca0, 0xc7c0c4c0,
            0xbf004100, 0xc7003c00, 0xc4c04840, 0xc7004100, 0xc7c04b40, 0xc7004400,
        ]);
    }
    if (result.halfMatrixWords !== null) {
        assert.deepEqual(result.halfMatrixWords, [
            ...new Uint32Array(new Float32Array([14, 19, 24, 6, 14, 32, 3, 6, 0, 2, 55, 12, 14, 32, 32, 77]).buffer),
            0x44004000, 0xbc004600, 0x4b804a00, 0xc0004c80,
            0x48004700, 0xc2004880, 0x49804900, 0xc4004a00,
            0x4d804d40, 0xc5004dc0, 0x52e04f80, 0xc6005080, 999,
        ]);
    }
    if (result.halfGeometryWords !== null) {
        const geometry = new Float32Array(new Uint32Array(result.halfGeometryWords).buffer);
        assert.deepEqual(Array.from(geometry, value => value === 0 ? 0 : value), [
            25, 5, 5, 5, 4, -3, 0, 4, 0, 1, 0, 0,
            0, 1, 0, -1, 0, 1, 0, -1, 0, 0, 0, 0, 60000, 1, 1, 60000,
        ]);
    }
    if (result.halfBitcastWords !== null) {
        assert.deepEqual(result.halfBitcastWords, [
            0xc0003c00, 0x04008000, 0xbc004400, 0x3c003800,
            0xc0003c00, 0xc0003c00, 0xbc004400, 0x3c003800,
            0xbc004400, 0x3c003800, 0xc0003c00, 0x04008000,
            0xc0003c00, 0x04008000, 0xc0003c00, 0x3c00c000,
            0x80000001, 0x4400c000, 0x00018000, 0x04003c00,
            0x7f7fffff, 0xff7fffff, 0xfbff7bff, 0x7f7fffff,
        ]);
    }
    if (result.halfNumericWords !== null) {
        const numeric = new Float32Array(new Uint32Array(result.halfNumericWords).buffer);
        const exact = [
            -2, -2, 2, 2, -1, -1, 1, 1, 0, 0, 1, 1, 0.5, 0.5, 0.5, 0.5,
            0, 3, 15, 17, 0.125, 0.5, 2, 2.25, 0.125, 1, 6, 9, 0, 1, 0, 0.31640625,
            0, 0.31640625, 1.5, 0.5, 3, 0.5, 3, 4,
        ];
        const approximate = [
            0.47942554, 0.87758256, 2.35619449, 1.44363548, 1.31695790, 0.54930614, 45, 3.14159265,
            1.64872127, 0.69314718, 3, -1, 0.54630249, 0.52359878, 1.04719755, 0.78539816,
            0.52109531, 1.12762597, 0.46211716, 2.5, -2, -3, -1, 0.5,
        ];
        assert.equal(numeric.length, exact.length + approximate.length);
        assert.deepEqual(Array.from(numeric.slice(0, exact.length), value => value === 0 ? 0 : value), exact);
        for (const [index, expected] of approximate.entries()) {
            const actual = numeric[index + exact.length];
            assert.ok(Math.abs(actual - expected) <= 0.004 * Math.max(1, Math.abs(expected)),
                      `Half numeric result ${index + exact.length} was ${actual}, expected ${expected}`);
        }
    }
    const quantized = result.quantizeWords.slice();
    if (quantized[6] === 0) quantized[6] = 0x33800000;
    assert.deepEqual(quantized, [
        0x3f800000, 0x3f804000, 0x80000000, 0x477fe000,
        0x38800000, 0x80000000, 0x33800000, 0xbf802000,
        0x40200000, 0x3eaaa000, 0xba832000, 0xc77fe000,
        0x3f800000, 0x3f804000, 0x40000000, 0x477fe000,
        0x38800000, 0, 0x80000000, 0x7f800000,
        0xff800000, 0x7f800000, 0xff800000, 0x7fc01234,
    ]);
    if (result.floatingVectorWords !== null) {
        const vector = new Float32Array(new Uint32Array(result.floatingVectorWords).buffer);
        assert.deepEqual(Array.from(vector, value => value === 0 ? 0 : value), [
            3.75, -2.75, 3, -3, -2.75, 3.75, -2, 2, -1.5, 1.5, 0, 0, -2.75, 1.5, 1, 0,
            -7, 19, -4, 40, -0.5, 0.5, 0, 0, 1, 0, 1, 0, 7.5, -5.5, 6, -6,
        ]);
    }
    assert.deepEqual(result.overrideWords, [3, 4, 0, 0, 8, 8, 8, 8, 7, 19, 3, 25]);
    assert.deepEqual(result.computedOverrideWords, [6, 7, 8, 0, 3, 3, 0, 0, 4, 8, 12, 16, 70, 71, 72]);
    assert.deepEqual(result.overrideArrayWords, [31, 31, 0, 0, 31, 33, 35, 33]);
    assert.deepEqual(result.pipelineBodyWords, [2, 3, 10, 0]);
    const aggregate = new Float32Array(Uint32Array.from(result.pipelineAggregateWords).buffer);
    assert.deepEqual(Array.from(aggregate), [3, 4, 7, 7, 22, 2, 3, 4]);
    assert.deepEqual(result.pipelineCompositeWords, [9, 18, 4, 4]);
    assert.deepEqual(result.arrayConstantWords, [
        10, 8, 3, 3, 0, 7,
        0, 0, 0, 23, 9, 22, 2, 23, 0xfffffffc, 34, 4, 23, 7, 0, 6, 23, 0, 0, 0, 23,
        14, 36, 33, 0x3f800000,
    ]);
    const pipelineBuiltins = new Float32Array(Uint32Array.from(result.pipelineBuiltinWords).buffer);
    assert.deepEqual(Array.from(pipelineBuiltins), [0.84375, -0.84375, 0.75, -0.75, 3, -6, 1, 0]);
    assert.deepEqual(result.conversionWords, [
        0x80000000, 0x7fffff80, 0xfffffffd, 0,
        0, 0xffffff00, 0, 0,
        1, 1, 1, 0,
        0, 1, 0x3f800000, 1,
        0xfffffffd, 9,
        0x80000000, 0x7fffff80, 0xfffffffd, 3, 0, 0xffffff00, 0, 3,
        0x7fffff80, 0x7fffff80, 0x7fffff80, 0x7fffff80, 0x7fffff80, 0x80000000, 0xffffff00, 0xffffff00,
        ...(result.halfConversionWords === null ? [] : [
            0xffff0020, 0xffe0, 0xfffffffe, 2, 0, 0xffe0, 0, 2,
            0xffe0, 0xffff0020, 0, 0, 0xffe0, 0, 0, 0,
        ]),
    ]);
    if (result.halfConversionWords !== null) {
        assert.deepEqual(result.halfConversionWords, [0, 0xfffffffe, 1, 0]);
    }
    assert.deepEqual(result.localArrayWords, [
        0, 0, 41, 101,
        0, 0, 42, 101,
        11, 19, 41, 102,
        0, 0, 41, 101,
    ]);
    assert.deepEqual(result.workgroupAggregateWords, [
        2, 31, 4, 35, 3, 32, 4, 35, 4, 33, 4, 35, 1, 34, 4, 35,
        12, 31, 4, 34, 13, 32, 4, 34, 14, 33, 4, 34, 11, 34, 4, 34,
    ]);
    assert.deepEqual(result.privateVariableWords, [
        30, 129, 0, 11, 30, 131, 42, 11, 30, 133, 43, 11, 30, 135, 0, 11,
        30, 137, 0, 11, 30, 139, 0, 11, 30, 141, 0, 11, 30, 143, 0, 11,
    ]);
    assert.deepEqual(result.privateRenderWords, [0xff0000ff, 0xff0000ff, 0xff0000ff, 0xff0000ff]);
    assert.deepEqual(result.pointerWords, [
        0, 0, 30, 219, 10, 11, 35, 219, 20, 21, 35, 219, 0, 0, 30, 219,
        40, 47, 147, 300, 41, 49, 149, 301, 42, 51, 151, 302, 43, 53, 153, 303,
    ]);
    assert.deepEqual(result.pointerSubviewWords, [
        50, 57, 1141, 320, 50, 60, 1144, 320, 20, 23, 1144, 312, 0, 0, 1141, 312,
        10, 40, 1420, 3246, 20, 30, 1124, 4040, 10, 40, 1620, 3250, 20, 30, 1126, 4440,
    ]);
    const sharedPointerGroup = [15, 126, 10, 126, 26, 126, 10, 127, 37, 126, 10, 128, 48, 126, 10, 129];
    assert.deepEqual(result.sharedPointerWords, [...sharedPointerGroup, ...sharedPointerGroup]);
    assert.deepEqual(result.swizzleAssignmentWords, [
        66, 55, 100, 10, 68, 59, 100, 10, 12, 12, 0, 0,
        11, 8, 4, 1, 0x3f800000, 0x40c00000, 0x40400000, 0x41000000, 0, 0, 1, 1,
    ]);
    assert.deepEqual(result.nestedStorageWords, [
        0, 0, 0, 2, 4, 5, 6, 103, 0, 0, 0, 204, 0, 0, 0, 2,
        7, 700, 701, 702,
        1, 2, 3, 703, 4, 5, 71, 704, 101, 705, 706, 707,
        11, 12, 13, 708, 14, 15, 16, 709, 202, 710, 711, 712,
        15, 900, 900, 900, 1, 9, 3, 900, 4, 5, 6, 900, 17, 900, 900, 900,
        24, 11, 40, 7,
    ]);
    if (result.nestedHalfStorageWords !== null) {
        assert.deepEqual(result.nestedHalfStorageWords, [0x44004000, 0xc7004880, 0x48004600, 0xc8004900]);
    }
    assert.deepEqual(result.nestedAtomicWords, [
        0, 0, 0, 0, 0, 0, 0, 2,
        5, 0xffffffec, 0xfffffff8, 98, 10, 12, 100, 2,
        0, 0, 0, 0, 20, 23, 200, 2,
        0, 0, 0, 0, 0, 0, 0, 2,
        7, 800, 0xfffffffc, 2, 801, 802, 34, 100, 803, 804,
        0xfffffffb, 0xfffffffa, 805, 806, 40, 200, 807, 808,
    ]);
    assert.deepEqual(result.nestedUniformWords, [
        0, 0, 3, 9, 4, 5, 9, 13, 0, 0, 3, 17, 0, 0, 3, 9,
        0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00,
        10, 0xc0000000, 0xbf800000, 0xbf800000, 31, 0xc0000000, 0xbf800000, 0xbf800000,
        59, 0xc1100000, 0x40e00000, 0xc0000000, 97, 0xc1100000, 0x40e00000, 0xc0000000,
        0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00,
    ]);
    if (result.nestedHalfUniformWords !== null) {
        const lanes = new Float32Array(Uint32Array.from(result.nestedHalfUniformWords).buffer);
        assert.deepEqual(Array.from(lanes), [1, -2, 3.5, 0, -4, 5, 6, 0]);
    }
    const standardLayout = await page.evaluate(() =>
        navigator.gpu.wgslLanguageFeatures?.has("uniform_buffer_standard_layout") ?? false);
    assert.equal(result.standardUniformWords !== null, standardLayout);
    if (standardLayout) {
        assert.deepEqual(result.standardUniformWords, [
            0, 0, 107, 12, 10, 20, 207, 12, 30, 40, 307, 12, 50, 60, 407, 12,
            0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00,
        ]);
    }
    assert.deepEqual(result.renderStorageBufferWords, [
        0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00,
        7, 400, 801, 10, 401, 802, 20, 402, 803, 30, 403, 804, 40,
    ]);
    const invalidDispatchCount = adapterLimits.dispatch === 0xffffffff ? 0 : adapterLimits.dispatch + 1;
    assert.deepEqual(result.indirectDispatchWords, [0, 5592, 0, 2, 3, 2, 1, invalidDispatchCount, 1, 1, 1, 0]);
    const firstInstance = Number(result.supportsIndirectFirstInstance);
    const indirectColor = firstInstance ? 0xff00ff00 : 0xff0000ff;
    assert.deepEqual(result.indirectDrawWords, [
        indirectColor, indirectColor, 0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000,
        firstInstance ? indirectColor : 0xffff0000, indirectColor,
        0, 3, 1, 1, firstInstance, 3, 1, 1, 0xffffffff, firstInstance, 3, 1, 2, 0xffffffff, 1,
        3, 0, 1, 1, 3, 1, 2, 1, 3, 1, 1, 2, 0xabcdef, 0x123456,
    ]);
    const supportsClipDistances = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("clip-distances"));
    assert.equal(result.clipDistancePixels !== null, supportsClipDistances);
    if (supportsClipDistances) {
        assert.deepEqual(result.clipDistancePixels, [
            0xffff0000, 0xff00ff00, 0xffff0000, 0xff00ff00, 0xffff0000, 0xff00ff00, 0xffff0000, 0xffff0000,
            0xffff0000, 0xff00ff00, 0xffff0000, 0xffff0000,
        ]);
    }
    const supportsPrimitiveIndex = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("primitive-index"));
    assert.equal(result.primitiveIndexPixels !== null, supportsPrimitiveIndex);
    if (supportsPrimitiveIndex) {
        assert.deepEqual(result.primitiveIndexPixels, [
            0xff000001, 0xff000002, 0xff000001, 0xff000002, 0xff000001, 0xff000002, 0xff000001, 0xff000002,
        ]);
    }
    const invariantRaster = [
        0xffff0000, 0xff00ff00, 0xff00ff00, 0xff00ff00,
        0xffff0000, 0xff00ff00, 0xff00ff00, 0xff00ff00,
        0xffff0000, 0xff00ff00, 0xff00ff00, 0xff00ff00,
        0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000,
    ];
    assert.deepEqual(result.invariantPositionPixels, [...invariantRaster, ...invariantRaster]);
    assert.deepEqual(result.voidFragmentPixels, [
        0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000,
        0xffff0000, 0xff00ff00, 0xffff0000, 0xff00ff00,
        0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000,
        0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000,
    ]);
    assert.deepEqual(result.discardContinuationWords, [
        0xffff0000, 0xff00ff00, 0xffff0000, 0xff00ff00, 3, 10, 3, 10,
    ]);
    const supportsDepthClipControl = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("depth-clip-control"));
    assert.equal(result.depthClipWords !== null, supportsDepthClipControl);
    if (supportsDepthClipControl) {
        assert.deepEqual(result.depthClipWords.slice(0, -1), [
            0xffff0000, 0x3f000000, 0xff00ff00, 0x3e800000, 0xff00ff00, 0x3f400000,
            0xffff0000, 0x3f000000, 0xffff0000, 0x3f000000, 0xff00ff00,
        ]);
        const interpolatedDepth = new Float32Array(new Uint32Array(result.depthClipWords.slice(-1)).buffer)[0];
        assert.ok(Math.abs(interpolatedDepth - 0.4375) <= 1 / 1024);
    }
    const supportsTimestampQuery = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("timestamp-query"));
    assert.equal(result.timestampQueryWords !== null, supportsTimestampQuery);
    if (supportsTimestampQuery) {
        const words = result.timestampQueryWords;
        assert.equal(words.length, 65);
        assert.deepEqual(words.slice(0, 13), [0x12345678, ...Array(12).fill(0)]);
        assert.deepEqual(words.slice(13, 26), words.slice(26, 39));
        assert.equal(words[13], 0x12345678);
        const samples = Array.from({ length: 6 }, (_, i) => BigInt(words[14 + 2 * i]) | (BigInt(words[15 + 2 * i]) << 32n));
        assert.equal(samples[0], 0n);
        assert.equal(samples[3], 0n);
        assert.ok(samples[1] > 0n && samples[2] >= samples[1]);
        assert.ok(samples[4] > 0n && samples[5] >= samples[4]);
        assert.equal(words[39], 0x12345678);
        const empty = Array.from({ length: 6 }, (_, i) => BigInt(words[40 + 2 * i]) | (BigInt(words[41 + 2 * i]) << 32n));
        assert.ok(empty[0] > 0n && empty[3] > 0n);
        assert.deepEqual([empty[1], empty[2], empty[4], empty[5]], [samples[1], samples[2], samples[4], samples[5]]);
        assert.equal(words[52], 0x12345678);
        const indirect = Array.from({ length: 6 }, (_, i) => BigInt(words[53 + 2 * i]) | (BigInt(words[54 + 2 * i]) << 32n));
        assert.ok(indirect[0] >= empty[0] && indirect[3] >= empty[3]);
        assert.deepEqual([indirect[1], indirect[2], indirect[4], indirect[5]], [samples[1], samples[2], samples[4], samples[5]]);
    }
    const occlusion = result.occlusionQueryWords;
    assert.equal(occlusion.length, supportsTimestampQuery ? 38 : 36);
    const visibility = Array.from({ length: 4 }, (_, pass) => {
        const base = pass * 9;
        assert.equal(occlusion[base], 0x11223344);
        return Array.from({ length: 4 }, (_, index) => Boolean(occlusion[base + 1 + index * 2] || occlusion[base + 2 + index * 2]));
    });
    assert.deepEqual(visibility, [
        [false, false, false, false], [false, true, false, true],
        [false, false, false, true], [false, false, false, true],
    ]);
    assert.deepEqual(occlusion.slice(18, 27), occlusion.slice(27, 36));
    if (supportsTimestampQuery) assert.ok((BigInt(occlusion[37]) << 32n | BigInt(occlusion[36])) > 0n);
    assert.deepEqual(result.mappedBufferWords, [8, 9, 10, 11, 0, 0, 0, 0, 9, 10, 11, 21, 22, 23, 24]);
    assert.deepEqual(result.bufferRangeWords, [
        0xee, 0xee, 0xee, 0xee, 5, 6, 7, 8, 0, 0, 0, 0,
        0xee, 0xee, 0xee, 0xee, 13, 14, 15, 16, 0, 0, 0, 0,
    ]);
    assert.deepEqual(result.queueTextureWords, [
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0xff112233, 0, 0xff445566, 0, 0xff778899, 0, 0xffaabbcc,
        0, 0, 0, 0,
        0, 0xff112233, 0, 0xff445566, 0, 0xff778899, 0, 0xffaabbcc,
        0x00f08001, 0x00a50711,
    ]);
    const supportsSubgroups = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("subgroups"));
    assert.equal(result.subgroupGatherWords !== null, supportsSubgroups);
    if (supportsSubgroups) {
        const subgroup = result.subgroupGatherWords;
        assert.equal(subgroup.length, 1080);
        for (let index = 0; index < 128; index++) {
            const start = index * 8;
            const [lane, size] = subgroup.slice(start, start + 2);
            assert.ok(size >= 4 && size <= 128 && lane < size);
            assert.deepEqual(subgroup.slice(start + 2, start + 8), [
                0, 0, lane ^ 1, size - 1 - lane, lane + 1 < size ? lane + 1 : lane, lane === 0 ? 0 : lane - 1,
            ]);
        }
        for (const word of subgroup.slice(1024, 1028)) {
            const size = word & 255, lane = word >>> 8 & 255, first = word >>> 16 & 255;
            assert.ok(word >>> 24 === 255 && size >= 4 && size <= 128 && lane < size && first <= lane);
        }
        assert.deepEqual(subgroup.slice(1028, 1044), [
            101, 3, 406, 0, 100, 3, 406, 1, 103, 3, 406, 2, 102, 3, 406, 3,
        ]);
        assert.deepEqual(subgroup.slice(1044, 1076), [
            101, 101, 1001, 0, 101, 101, 1001, 1, 101, 101, 1001, 2, 101, 101, 1001, 3,
            102, 102, 1002, 0, 102, 102, 1002, 1, 102, 102, 1002, 2, 102, 102, 1002, 3,
        ]);
        for (const word of subgroup.slice(1076)) {
            const size = word & 255, lane = word >>> 8 & 255, first = word >>> 16 & 255;
            assert.ok(word >>> 24 === 173 && size >= 4 && size <= 128 && (size & (size - 1)) === 0);
            assert.ok(first <= lane && lane < size);
        }
    }
    assert.equal(result.subgroupCollectiveWords !== null, supportsSubgroups);
    if (supportsSubgroups) {
        const words = result.subgroupCollectiveWords;
        assert.equal(words.length, 3660);
        for (let index = 0; index < 128; index++) {
            const base = index * 28;
            const [size, lane] = words.slice(base, base + 2);
            assert.ok(size >= 4 && size <= 128 && (size & (size - 1)) === 0 && lane < size);
            const ballot = Array.from({ length: 4 }, (_, word) => {
                const bits = Math.min(32, Math.max(0, size - word * 32));
                return 2 ** bits - 1;
            });
            assert.deepEqual(words.slice(base, base + 28), [
                size, lane, 1, 1, ...ballot.map(mask => mask & 0x55555555),
                size * (size - 1) / 2, 8, 7, size - 1,
                0x80000000, size - 1, 0, lane * (lane + 1) / 2,
                lane === 0 ? 0 : lane * (lane - 1) / 2, 1 << Math.min(lane, 3), 1 << Math.min(lane + 1, 3), lane === 0 ? 1 : 0,
                ...ballot, 1, 1, 1, 1,
            ]);
        }
        assert.deepEqual(words.slice(3584, 3588), [0x03020101, 0x02030001, 0x01000301, 0x00010201]);
        assert.deepEqual(words.slice(3588), [
            0, 0, 0, 0, 0, 0, 0, 0,  1, 0, 1, 0, 1, 0, 0, 0,  1, 1, 1, 0, 1, 0, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1,  1, 0, 1, 1, 1, 1, 1, 1,  0, 0, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,  1, 0, 0, 1, 1, 1, 1, 1,  0, 0, 1, 0, 0, 1, 1, 1,
        ]);
    }
    const supportsSubgroupID = supportsSubgroups && await page.evaluate(() =>
        navigator.gpu.wgslLanguageFeatures.has("subgroup_id"));
    assert.equal(result.subgroupIndexWords !== null, supportsSubgroupID);
    if (supportsSubgroupID) {
        const words = result.subgroupIndexWords;
        assert.equal(words.length, 1040);
        for (let group = 0; group < 2; group++) {
            const base = group * 520;
            const count = words[base + 1], size = words[base + 3];
            assert.ok(count >= 2 && count <= 130 && size >= 4 && size <= 128);
            const lanes = Array.from({ length: count }, () => new Set());
            for (let i = base; i < base + 520; i += 4) {
                const id = words[i], lane = words[i + 2];
                assert.equal(words[i + 1], count);
                assert.equal(words[i + 3], size);
                assert.ok(id < count && lane < size && !lanes[id].has(lane));
                lanes[id].add(lane);
            }
            assert.ok(lanes.every(groupLanes => groupLanes.size > 0));
            assert.ok(lanes.some(groupLanes => groupLanes.size < size));
        }
    }
    const supportsDualSource = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("dual-source-blending"));
    assert.equal(result.dualSourcePixels !== null, supportsDualSource);
    if (supportsDualSource) {
        const expected = [[128, 223, 48, 64], [128, 159, 16, 191], [255, 128, 64, 255]];
        assert.equal(result.dualSourcePixels.length, expected.length);
        for (const [index, pixel] of result.dualSourcePixels.entries()) {
            for (let channel = 0; channel < 4; channel++) {
                assert.ok(Math.abs((pixel >>> (8 * channel) & 255) - expected[index][channel]) <= 1);
            }
        }
    }
    const supportsBC = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("texture-compression-bc"));
    assert.equal(result.bcTextureWords !== null, supportsBC);
    if (supportsBC) {
        assert.deepEqual(result.bcTextureWords.slice(0, 16), [
            0, 0, 0, 0, 0, 0, 0, 0, 0x7bef7bef, 0, 0x7bef7bef, 0, 0x1234ffff, 0, 0x7bef7bef, 0,
        ]);
        const pixels = new Float32Array(new Uint32Array(result.bcTextureWords.slice(16)).buffer);
        assert.equal(pixels.length, 4);
        assert.ok(pixels[0] > 0.46 && pixels[0] < 0.50);
        assert.ok(pixels[1] > 0.17 && pixels[1] < 0.22);
        assert.deepEqual(Array.from(pixels.slice(2)), [1, 1]);
    }
    const supportsETC2 = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("texture-compression-etc2"));
    assert.equal(result.etc2TextureWords !== null, supportsETC2);
    if (supportsETC2) {
        assert.deepEqual(result.etc2TextureWords.slice(0, 8), [0x00aaaaaa, 0, 0x00666666, 0, 0x80, 0, 0x40, 0]);
        const pixels = new Float32Array(new Uint32Array(result.etc2TextureWords.slice(8)).buffer);
        assert.equal(pixels.length, 4);
        for (const [index, [min, max]] of [[0.40, 0.42], [0.13, 0.15], [0.48, 0.52], [0.23, 0.27]].entries()) {
            assert.ok(pixels[index] >= min && pixels[index] <= max);
        }
    }
    const supportsASTC = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("texture-compression-astc"));
    assert.equal(result.astcTextureWords !== null, supportsASTC);
    if (supportsASTC) {
        assert.deepEqual(result.astcTextureWords.slice(0, 20), [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xfffffdfc, 0xffffffff, 0x80808080, 0xffff8080,
            0xfffffdfc, 0xffffffff, 0x0000bfbf, 0xffff0000,
        ]);
        const pixels = new Float32Array(new Uint32Array(result.astcTextureWords.slice(20)).buffer);
        assert.equal(pixels.length, 4);
        for (const [index, [min, max]] of [[0.49, 0.51], [0.20, 0.23], [0.74, 0.76], [1, 1]].entries()) {
            assert.ok(pixels[index] >= min && pixels[index] <= max);
        }
    }
    const [supportsBCVolume, supportsASTCVolume] = await page.evaluate(async () => {
        const features = (await navigator.gpu.requestAdapter()).features;
        return ["texture-compression-bc-sliced-3d", "texture-compression-astc-sliced-3d"].map(feature => features.has(feature));
    });
    for (const [supported, words, blocks] of [
        [supportsBCVolume, result.bcVolumeWords, [[0x40004000, 0], [0x80008000, 0]]],
        [supportsASTCVolume, result.astcVolumeWords, [[0xfffffdfc, 0xffffffff, 0x4040, 0xffff0000],
            [0xfffffdfc, 0xffffffff, 0x8080, 0xffff0000]]],
    ]) {
        assert.equal(words !== null, supported);
        if (!supported) continue;
        const blockWords = blocks[0].length;
        const expected = new Array(12 * blockWords).fill(0);
        expected.splice(7 * blockWords, blockWords, ...blocks[0]);
        expected.splice(11 * blockWords, blockWords, ...blocks[1]);
        assert.deepEqual(words.slice(0, expected.length), expected);
        const pixels = new Float32Array(new Uint32Array(words.slice(expected.length)).buffer);
        assert.equal(pixels.length, 4);
        for (const [index, value] of [0.25, 0.5, 0.75, 1].entries()) {
            assert.ok(Math.abs(pixels[index] - value) < 0.04);
        }
    }
    assert.deepEqual(result.callStatementWords, [23, 16, 2, 11, 7, 23, 17, 5, 11, 23, 17, 5]);
    assert.deepEqual(result.matrixCompoundWords.slice(0, 3), [1, 1, 1]);
    const matrixCompoundValues = new Float32Array(new Uint32Array(result.matrixCompoundWords.slice(3)).buffer);
    assert.deepEqual(Array.from(matrixCompoundValues), [14, 20, 26, 32, 40, 48, 1, 4, 8, 9, 8]);
    const supportsFloat32Filtering = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("float32-filterable"));
    const float32Values = new Float32Array(new Uint32Array(result.float32TextureWords.slice(0, -1)).buffer);
    assert.deepEqual(Array.from(float32Values), [
        0, 0, 0, 2, 1, 10, 0.25, 2, ...(supportsFloat32Filtering ? [2, 12, 0.5, 2] : []), 3, 14, 0.75, 2,
    ]);
    assert.equal(result.float32TextureWords.at(-1), 0xff00bf00);
    const supportsFloat32Blending = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("float32-blendable"));
    const float32Colors = new Float32Array(new Uint32Array(result.float32ColorWords).buffer);
    const multisampleFloat32 = mask => [[2.5, 0, 0, 4], [-1, 3, 0, 4], [0.5, -0.5, 4, 4]]
        .filter((_, index) => mask & (1 << index));
    const supportedMultisamples = multisampleFloat32(result.multisampledFloat32Mask);
    assert.deepEqual(Array.from(float32Colors), [
        2.5, -1, 3, 0.5, -0.5, 4, 0.25,
        ...(supportsFloat32Blending ? [3.5, 0, 1, 1.5, -2.5, 7, 0.75] : []),
        ...supportedMultisamples.flat(),
    ]);
    assert.deepEqual(result.halfTexelWords, [0x00001234, 0x42001234, 0x3c001234, 0x40000000]);
    assert.deepEqual(result.halfColorWords, [
        0x43000000, 0x3c000000, 0xc1003e00, 0x3a004700,
        ...(result.halfPrecisionWords === null ? [] : [0x3400, 0x3a003400, 0x3c003a00, 0x3400, 0x3a003400, 0x3c003a00]),
    ]);
    const [supportsTier1Storage, supportsTier2StorageReadWrite, supportsNormalized16] = await page.evaluate(async () => {
        const features = (await navigator.gpu.requestAdapter()).features;
        const readable = navigator.gpu.wgslLanguageFeatures.has("readonly_and_readwrite_storage_textures");
        return [readable && features.has("texture-formats-tier1"), readable && features.has("texture-formats-tier2"),
            features.has("texture-formats-tier1")];
    });
    const halfStorageValues = [0.5, -2, 0.25, 1, 1.5, -1, 1.25, 1];
    const updatedHalfStorageValues = [2.5, -2, 0.25, 1, 3.5, -1, 1.25, 1];
    const decodeFloatWords = words => Array.from(new Float32Array(new Uint32Array(words).buffer));
    assert.deepEqual(decodeFloatWords(result.halfStorageWords), [
        ...(supportsTier1Storage ? halfStorageValues : []), ...(supportsTier2StorageReadWrite ? updatedHalfStorageValues : []),
    ]);
    const integer16Texels = [
        60000, 0, 0, 1, 65535, 300, 0, 1, 1, 257, 65534, 1234,
        -20000, 0, 0, 1, -32768, 300, 0, 1, -1, 0, 32767, -1234,
    ].map(value => value >>> 0);
    const integer16StorageTexels = [...integer16Texels.slice(0, 8), ...integer16Texels.slice(12, 20)];
    assert.deepEqual(result.integer16ColorWords, [0xea600000, 0xb1e00000, ...integer16Texels]);
    assert.deepEqual(result.integer16MSAAWords, integer16Texels);
    assert.deepEqual(result.integer16StorageWords, [
        ...(supportsTier1Storage ? integer16StorageTexels : []),
        ...(supportsTier2StorageReadWrite ? [0xee480000, 0xadf80000, 0x00020001, 0x00040003, 0xfffeffff, 0xfffcfffd] : []),
    ]);
    const normalized16Values = [
        1, 0, 0, 1, 0.25, 0.75, 0, 1, 0, 0.25, 0.75, 1,
        -0.5, 0, 0, 1, -1, 1, 0, 1, -1, -0.25, 0.25, 1,
    ];
    if (supportsTier1Storage) {
        const [packed, ...words] = result.normalized16StorageWords;
        assert.equal(packed, 0xffff0000);
        const actual = decodeFloatWords(words);
        assert.equal(actual.length, 2 * normalized16Values.length);
        for (const [index, value] of actual.entries()) {
            assert.ok(Math.abs(value - normalized16Values[index % normalized16Values.length]) < 0.0001);
        }
    } else assert.deepEqual(result.normalized16StorageWords, []);
    if (supportsNormalized16) {
        assert.equal(result.normalized16ColorWords.length, 4);
        const [red, rg, rgLow, rgHigh] = result.normalized16ColorWords;
        const signed = word => (word << 16) >> 16;
        const actual = [red >>> 16, signed(rg), signed(rg >>> 16), rgLow & 65535, rgLow >>> 16, rgHigh & 65535, rgHigh >>> 16];
        for (const [index, reference] of [49151, -8192, 24575, 32768, 32768, 65535, 49151].entries()) {
            assert.ok(Math.abs(actual[index] - reference) <= 1);
        }
    } else assert.deepEqual(result.normalized16ColorWords, []);
    const supportsRG11B10Render = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("rg11b10ufloat-renderable"));
    const packedCoreWords = [0xbff003ff, 0x803803ff];
    const hdrSampleWords = [0, 0, 0, 0x3f800000, 0x3f800000, 0x40000000, 0x40800000, 0x3f800000];
    assert.deepEqual(result.packedColorWords, [
        ...packedCoreWords, ...(supportsRG11B10Render ? [0x882003d0] : []), ...hdrSampleWords,
    ]);
    const packedStorageExpected = [
        ...packedCoreWords, 0x882003c0, 0x3f800000, 0, 0x3f800000, 0x3f2aaaab,
        1023, 512, 3, 2, 0x3f800000, 0x40000000, 0x40800000, 0x3f800000,
    ];
    assert.deepEqual(result.packedStorageWords, supportsTier1Storage ? packedStorageExpected : []);
    assert.equal(result.srgbWords.length, 10);
    assert.deepEqual(result.srgbWords.slice(0, 4), [0x80204080, 0x80204080, 0x80804020, 0x80804020]);
    const decodedSRGB = decodeFloatWords(result.srgbWords.slice(4, 8));
    for (const [index, value] of [0.21586, 0.05127, 128 / 255, 128 / 255].entries()) {
        assert.ok(Math.abs(decodedSRGB[index] - value) < 0.001);
    }
    for (const [index, expected] of [[137, 188, 225, 128], [225, 188, 137, 128]].entries()) {
        const word = result.srgbWords[8 + index];
        for (const [lane, value] of expected.entries()) assert.ok(Math.abs((word >>> (8 * lane) & 255) - value) <= 1);
    }
    assert.equal(result.sharedExponentWords.length, 10);
    assert.deepEqual(decodeFloatWords(result.sharedExponentWords.slice(0, 4)), [0, 0, 0, 1]);
    assert.deepEqual(result.sharedExponentWords.slice(4, 6), [0x94050241, 0xa5054221]);
    assert.deepEqual(decodeFloatWords(result.sharedExponentWords.slice(6)), [1.5390625, 6.0390625, 12.0390625, 1]);
    const expectedByteSamples = [
        128 / 255, 0, 0, 1, 64 / 255, 192 / 255, 0, 1,
        -64 / 127, 0, 0, 1, -63 / 127, 1, 0, 1,
        250, 0, 0, 1, 4, 200, 0, 1, -127, 0, 0, 1, -128, 100, 0, 1,
    ];
    const assertByteSamples = words => {
        const actual = decodeFloatWords(words);
        assert.equal(actual.length, expectedByteSamples.length);
        for (const [index, value] of expectedByteSamples.entries()) {
            assert.ok(Math.abs(actual[index] - value) < 0.0001);
        }
    };
    assert.equal(result.byteTransferWords[0], 0xaabb80dd);
    assertByteSamples(result.byteTransferWords.slice(1));
    const unormByteColors = [0x80000000, 0xc0400000, 0xff0000ff];
    assert.deepEqual(result.byteColorWords, [
        ...unormByteColors, ...(supportsNormalized16 ? [0xc0000000, 0x7fc10000, 0x7f007fc0] : []),
    ]);
    if (supportsTier1Storage) {
        assertByteSamples(result.byteStorageWords.slice(0, 32));
        assert.deepEqual(result.byteStorageWords.slice(32), supportsTier2StorageReadWrite ? [0xfd000000, 0xff0000ff] : []);
    } else assert.deepEqual(result.byteStorageWords, []);
    assert.equal(result.depthOnlyFormatWords.length, 7);
    assert.deepEqual(result.depthOnlyFormatWords.slice(0, 3), [0xbfff4000, 0xffff8000, 0x40004000]);
    for (const [index, actual] of decodeFloatWords(result.depthOnlyFormatWords.slice(3)).entries()) {
        assert.ok(Math.abs(actual - [0.25, 0.25, 0.25, 1][index]) < 0.0001);
    }
    const supportsDepth32Stencil = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("depth32float-stencil8"));
    assert.deepEqual(result.depth32StencilWords, supportsDepth32Stencil ?
        [0x0703, 0xff00ff00, 0, 0x3e800000, 0x3f400000, 0x0704, 0x3e800000, 0x3f400000, 0x40800000, 0x40e00000] : []);
    const stencilOnlyExpected = [0, 0x0703, 0xff00ff00, 0, 0x0704, 4, 7, 4, 7];
    assert.deepEqual(result.stencilOnlyWords, stencilOnlyExpected);
    const supportsBGRAStorage = await page.evaluate(async () =>
        (await navigator.gpu.requestAdapter()).features.has("bgra8unorm-storage"));
    if (supportsBGRAStorage) {
        assert.equal(result.bgraStorageWords.length, 5);
        assert.equal(result.bgraStorageWords[0], 0xff4080bf);
        const values = decodeFloatWords(result.bgraStorageWords.slice(1));
        for (const [index, expected] of [64 / 255, 128 / 255, 191 / 255, 1].entries()) {
            assert.ok(Math.abs(values[index] - expected) < 0.0001);
        }
    } else assert.deepEqual(result.bgraStorageWords, []);
    const wideStorageExpected = [
        0x3c003e00, 0x40004500,
        ...new Uint32Array(new Float32Array([-1.5, 2.25, -3.75, 5.25]).buffer),
        101, 202, 303, 404, ...[-11, 22, -33, 44].map(value => value >>> 0),
    ];
    assert.deepEqual(result.wideStorageWords, supportsTier2StorageReadWrite ? wideStorageExpected : []);
    assert.equal(result.forwardedResourceWords.length, 22);
    assert.deepEqual(decodeFloatWords(result.forwardedResourceWords.slice(0, 4)).map(value => Math.round(value * 255)),
        [70, 10, 220, 150]);
    assert.deepEqual(result.forwardedResourceWords.slice(4, 8), [0xff000046, 0xff0000dc, 0xff00ff46, 0xff00ff96]);
    assert.deepEqual(decodeFloatWords(result.forwardedResourceWords.slice(8, 12)).map(value => Math.round(value * 255)),
        [150, 220, 10, 70]);
    assert.deepEqual(result.forwardedResourceWords.slice(12, 14), [0xff00ff0a, 0xff00ff96]);
    assert.deepEqual(decodeFloatWords(result.forwardedResourceWords.slice(14, 18)).map(value => Math.round(value * 255)),
        [210, 210, 170, 170]);
    assert.deepEqual(decodeFloatWords(result.forwardedResourceWords.slice(18)).map(value => Math.round(value * 255)),
        [660, 660, 310, 310]);
    assert.deepEqual(result.webBufferArrayWords, [
        0, 0, 0, 0, 100, 101, 1, 11, 200, 201, 3, 22, 0, 0, 0, 0,
        101, 55, 41, 42, 201, 77, 8, 9, 0xff37010b, 0xff4d030b,
        1, 55, 55, 112, 3, 77, 9, 223, 101, 110, 41, 42, 201, 86, 8, 9,
        0, 1, 0, 1132, 1, 3, 132, 1086, 3, 0, 86, 1000, 0, 0, 0, 1000,
        101, 110, 41, 42, 201, 86, 8, 9,
        0, 0, 0, 0, 0, 1, 0, 1, 17, 110, 41, 42, 23, 86, 8, 9,
    ]);
    assert.equal(result.externalTextureWords.length, 18);
    const externalExpected = [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 1, 2, 1];
    for (const [index, value] of decodeFloatWords(result.externalTextureWords.slice(0, 16)).entries()) {
        assert.ok(Math.abs(value - externalExpected[index]) < 0.02);
    }
    assert.deepEqual(result.externalTextureWords.slice(16), [0xff0000ff, 0xff00ff00]);
    assert.deepEqual(result.immediateWords, result.maxImmediateSize >= 32
        ? [0x11223344, 0xaabbccdd, 0x81234567, 9, 21, 22, 23, 0] : []);
    assert.deepEqual(result.renderImmediateWords, result.maxImmediateSize >= 32
        ? [0xff0000ff, 0xff00ff00, 0xffff0000, 0xff00ffff] : []);
    assert.equal(result.linearIndexingInvocations, result.supportsLinearIndexing ? 288 : 0);
    assert.equal(result.requestedSubgroupSize, result.supportsSubgroupSizeControl ? result.subgroupMinSize : 0);
    const subgroupUniformity = result.supportsSubgroupUniformity && supportsSubgroupID;
    assert.equal(result.subgroupUniformityWords.length, subgroupUniformity ? 2 : 0);
    if (subgroupUniformity) {
        const [size, groups] = result.subgroupUniformityWords;
        assert.ok(size > 0 && size <= 128 && groups === Math.ceil(128 / size));
    }
    assert.equal(result.submissions, (result.halfPrecisionWords === null ? 140 : 150) +
        Number(standardLayout) + Number(supportsClipDistances) + Number(supportsPrimitiveIndex) +
        Number(supportsDepthClipControl) + 4 * Number(supportsTimestampQuery) + 2 * Number(supportsSubgroups) + Number(supportsSubgroupID) + Number(supportsDualSource) + Number(supportsBC) + Number(supportsETC2) + Number(supportsASTC) +
        Number(supportsBCVolume) + Number(supportsASTCVolume) + Number(supportsFloat32Filtering) + Number(supportsFloat32Blending) + supportedMultisamples.length + 5 * Number(supportsTier1Storage) + Number(supportsNormalized16) + Number(supportsDepth32Stencil) +
        Number(supportsBGRAStorage) + Number(supportsTier2StorageReadWrite) + 2 * Number(result.maxImmediateSize >= 32) +
        Number(result.supportsLinearIndexing) + Number(result.requestedSubgroupSize > 0) + Number(subgroupUniformity));
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
        const requestDevice = GPUAdapter.prototype.requestDevice;
        let disableTier1 = false;
        GPUAdapter.prototype.requestDevice = function (descriptor) {
            const requiredFeatures = descriptor.requiredFeatures.filter(feature => !["indirect-first-instance", "clip-distances", "primitive-index", "depth-clip-control", "timestamp-query", "subgroups", "subgroup-size-control", "dual-source-blending", "texture-compression-bc", "texture-compression-etc2", "texture-compression-astc", "texture-compression-bc-sliced-3d", "texture-compression-astc-sliced-3d", "float32-filterable", "float32-blendable", "texture-formats-tier2", "depth32float-stencil8", "bgra8unorm-storage"].includes(feature) &&
                (!disableTier1 || !["texture-formats-tier1", "rg11b10ufloat-renderable"].includes(feature)));
            return requestDevice.call(this, { ...descriptor, requiredFeatures });
        };
        try {
            const offscreen = await runDiagnostic();
            disableTier1 = true;
            const withoutTier1 = await runDiagnostic();
            return {
                duplicate, cancelled, retryFrames: retry.presentedFrames, offscreenFrames: offscreen.presentedFrames,
                offscreenFirstInstance: offscreen.supportsIndirectFirstInstance,
                offscreenIndirectPixels: offscreen.indirectDrawWords.slice(0, 8),
                offscreenClipDistancePixels: offscreen.clipDistancePixels,
                offscreenPrimitiveIndexPixels: offscreen.primitiveIndexPixels,
                offscreenDepthClipWords: offscreen.depthClipWords,
                offscreenTimestampQueryWords: offscreen.timestampQueryWords,
                offscreenSubgroupGatherWords: offscreen.subgroupGatherWords,
                offscreenSubgroupIndexWords: offscreen.subgroupIndexWords,
                offscreenRequestedSubgroupSize: offscreen.requestedSubgroupSize,
                offscreenDualSourcePixels: offscreen.dualSourcePixels,
                offscreenBCTextureWords: offscreen.bcTextureWords,
                offscreenETC2TextureWords: offscreen.etc2TextureWords,
                offscreenASTCTextureWords: offscreen.astcTextureWords,
                offscreenBCVolumeWords: offscreen.bcVolumeWords,
                offscreenASTCVolumeWords: offscreen.astcVolumeWords,
                offscreenFloat32TextureWords: offscreen.float32TextureWords,
                offscreenFloat32ColorWords: offscreen.float32ColorWords,
                offscreenMultisampledFloat32Mask: offscreen.multisampledFloat32Mask,
                offscreenHalfStorageWords: offscreen.halfStorageWords,
                noTier1HalfStorageWords: withoutTier1.halfStorageWords,
                offscreenInteger16StorageWords: offscreen.integer16StorageWords,
                noTier1Integer16StorageWords: withoutTier1.integer16StorageWords,
                noTier1Integer16ColorWords: withoutTier1.integer16ColorWords,
                noTier1Integer16MSAAWords: withoutTier1.integer16MSAAWords,
                noTier1Normalized16StorageWords: withoutTier1.normalized16StorageWords,
                noTier1Normalized16ColorWords: withoutTier1.normalized16ColorWords,
                noTier1PackedColorWords: withoutTier1.packedColorWords,
                noTier1PackedStorageWords: withoutTier1.packedStorageWords,
                noTier1ByteColorWords: withoutTier1.byteColorWords,
                noTier1ByteStorageWords: withoutTier1.byteStorageWords,
                offscreenByteStorageWords: offscreen.byteStorageWords,
                offscreenDepth32StencilWords: offscreen.depth32StencilWords,
                offscreenStencilOnlyWords: offscreen.stencilOnlyWords,
                offscreenBGRAStorageWords: offscreen.bgraStorageWords,
                offscreenWideStorageWords: offscreen.wideStorageWords,
            };
        } finally {
            GPUAdapter.prototype.requestDevice = requestDevice;
        }
    });
    assert.match(lifecycle.duplicate, /Canvas already belongs to a diagnostic/);
    assert.match(lifecycle.cancelled, /Canvas diagnostic closed/);
    assert.equal(lifecycle.retryFrames, 2);
    assert.equal(lifecycle.offscreenFrames, 0);
    assert.equal(lifecycle.offscreenFirstInstance, false);
    assert.equal(lifecycle.offscreenClipDistancePixels, null);
    assert.equal(lifecycle.offscreenPrimitiveIndexPixels, null);
    assert.equal(lifecycle.offscreenDepthClipWords, null);
    assert.equal(lifecycle.offscreenTimestampQueryWords, null);
    assert.equal(lifecycle.offscreenSubgroupGatherWords, null);
    assert.equal(lifecycle.offscreenSubgroupIndexWords, null);
    assert.equal(lifecycle.offscreenRequestedSubgroupSize, 0);
    assert.equal(lifecycle.offscreenDualSourcePixels, null);
    assert.equal(lifecycle.offscreenBCTextureWords, null);
    assert.equal(lifecycle.offscreenETC2TextureWords, null);
    assert.equal(lifecycle.offscreenASTCTextureWords, null);
    assert.equal(lifecycle.offscreenBCVolumeWords, null);
    assert.equal(lifecycle.offscreenASTCVolumeWords, null);
    assert.deepEqual(Array.from(new Float32Array(new Uint32Array(lifecycle.offscreenFloat32TextureWords.slice(0, -1)).buffer)),
        [0, 0, 0, 2, 1, 10, 0.25, 2, 3, 14, 0.75, 2]);
    assert.equal(lifecycle.offscreenFloat32TextureWords.at(-1), 0xff00bf00);
    assert.deepEqual(Array.from(new Float32Array(new Uint32Array(lifecycle.offscreenFloat32ColorWords).buffer)),
        [2.5, -1, 3, 0.5, -0.5, 4, 0.25, ...multisampleFloat32(lifecycle.offscreenMultisampledFloat32Mask).flat()]);
    assert.deepEqual(decodeFloatWords(lifecycle.offscreenHalfStorageWords), supportsTier1Storage ? halfStorageValues : []);
    assert.deepEqual(lifecycle.noTier1HalfStorageWords, []);
    assert.deepEqual(lifecycle.offscreenInteger16StorageWords, supportsTier1Storage ? integer16StorageTexels : []);
    assert.deepEqual(lifecycle.noTier1Integer16StorageWords, []);
    assert.deepEqual(lifecycle.noTier1Integer16ColorWords, [0xea600000, 0xb1e00000, ...integer16Texels]);
    assert.deepEqual(lifecycle.noTier1Integer16MSAAWords, integer16Texels);
    assert.deepEqual(lifecycle.noTier1Normalized16StorageWords, []);
    assert.deepEqual(lifecycle.noTier1Normalized16ColorWords, []);
    assert.deepEqual(lifecycle.noTier1PackedColorWords, [...packedCoreWords, ...hdrSampleWords]);
    assert.deepEqual(lifecycle.noTier1PackedStorageWords, []);
    assert.deepEqual(lifecycle.noTier1ByteColorWords, unormByteColors);
    assert.deepEqual(lifecycle.noTier1ByteStorageWords, []);
    if (supportsTier1Storage) assertByteSamples(lifecycle.offscreenByteStorageWords);
    else assert.deepEqual(lifecycle.offscreenByteStorageWords, []);
    assert.deepEqual(lifecycle.offscreenDepth32StencilWords, []);
    assert.deepEqual(lifecycle.offscreenStencilOnlyWords, stencilOnlyExpected);
    assert.deepEqual(lifecycle.offscreenBGRAStorageWords, []);
    assert.deepEqual(lifecycle.offscreenWideStorageWords, []);
    assert.deepEqual(lifecycle.offscreenIndirectPixels, [
        0xff0000ff, 0xff0000ff, 0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000, 0xff0000ff,
    ]);
    const incomplete = await page.evaluate(async () => {
        const { openWebGPU } = await import("/webgpu.mjs");
        const host = await openWebGPU();
        try {
            const stage = new Uint8Array(48);
            const header = new DataView(stage.buffer);
            header.setUint32(0, 23, true);
            header.setBigUint64(4, 1n, true);
            header.setBigUint64(12, 1n, true);
            header.setBigUint64(20, BigInt(16 * 1024 * 1024 + 4), true);
            header.setUint32(36, 8, true);
            header.setUint32(40, 6, true);
            await host.handle(stage);
            const execute = stage.slice(0, 20);
            new DataView(execute.buffer).setUint32(0, 24, true);
            const error = await host.handle(execute).then(() => "unexpected success", failure => failure.message);
            return { error, ...host.metrics() };
        } finally {
            host.close();
        }
    });
    assert.match(incomplete.error, /Incomplete staged browser submission/);
    assert.equal(incomplete.submissions, 0);
    assert.equal(incomplete.remainingResources, 0);
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
