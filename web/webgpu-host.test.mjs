import assert from "node:assert/strict";
import { test } from "node:test";
import { openWebGPU } from "./webgpu.mjs";

test("deferred presentation can attach the engine canvas after opening the host", async () => {
    globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 16, TEXTURE_BINDING: 4 };
    const device = {
        createTexture() { return { destroy() {} }; },
        features: new Set(), limits: { maxSampledTexturesPerShaderStage: 16 }, lost: new Promise(() => {}),
        pushErrorScope() {}, async popErrorScope() { return null; },
        createShaderModule() {}, async createRenderPipelineAsync() {}, destroy() {},
    };
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: {
        gpu: { requestAdapter: async () => ({
            info: { isFallbackAdapter: false }, features: new Set(), limits: { maxSampledTexturesPerShaderStage: 32 },
            requestDevice: async () => device,
        }) },
    } });
    const headless = await openWebGPU(null);
    const deferred = await openWebGPU(null, { deferCanvas: true });
    let calls = 0;
    const context = {};
    const canvas = { getContext: name => { assert.equal(name, "webgpu"); calls++; return context; } };
    try {
        assert.equal(headless.maxSampledTexturesPerShaderStage, 16);
        assert.equal(headless.supportsPresentation, false);
        assert.equal(deferred.supportsPresentation, true);
        assert.throws(() => deferred.attachCanvas({ getContext: () => null }), /cannot supply/);
        deferred.attachCanvas(canvas);
        deferred.attachCanvas(canvas);
        assert.equal(calls, 1);
        assert.equal(deferred.metrics().surfaceSize, null);
    } finally {
        headless.close();
        deferred.close();
        delete globalThis.navigator;
        delete globalThis.GPUTextureUsage;
    }
});


test("invalid device limits reject and release the created device", async () => {
    let destroyed = 0;
    const device = { limits: {}, destroy() { destroyed++; } };
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: {
        requestAdapter: async () => ({ info: {}, features: new Set(), limits: { maxSampledTexturesPerShaderStage: 32 },
            requestDevice: async () => device }),
    } } });
    try {
        await assert.rejects(openWebGPU(null), /Invalid WebGPU device sampled texture limit/);
        assert.equal(destroyed, 1);
    } finally { delete globalThis.navigator; }
});
