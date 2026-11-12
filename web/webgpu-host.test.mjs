import assert from "node:assert/strict";
import { test } from "node:test";
import { openWebGPU } from "./webgpu.mjs";

test("deferred presentation can attach the engine canvas after opening the host", async () => {
    globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 16, TEXTURE_BINDING: 4 };
    const device = {
        createTexture() { return { destroy() {} }; },
        features: new Set(), limits: {}, lost: new Promise(() => {}),
        pushErrorScope() {}, async popErrorScope() { return null; },
        createShaderModule() {}, async createRenderPipelineAsync() {}, destroy() {},
    };
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: {
        gpu: { requestAdapter: async () => ({
            info: { isFallbackAdapter: false }, features: new Set(), limits: {},
            requestDevice: async () => device,
        }) },
    } });
    const headless = await openWebGPU(null);
    const deferred = await openWebGPU(null, { deferCanvas: true });
    let calls = 0;
    const context = {};
    const canvas = { getContext: name => { assert.equal(name, "webgpu"); calls++; return context; } };
    try {
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
