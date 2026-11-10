import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
const limits = new Proxy({ maxBufferSize: 1073741824, maxTextureDimension2D: 16384 },
    { get: (target, key) => target[key] ?? 256 });
let requested;
const device = {
    limits: new Proxy({}, { get: (_, key) => requested.requiredLimits[key] ?? 256 }),
    features: new Set(), lost: new Promise(() => {}),
    pushErrorScope() {}, popErrorScope: async () => null, createTexture: () => ({ destroy() {} }),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}), destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: {
    requestAdapter: async () => ({ info: { isFallbackAdapter: false }, limits, features: new Set(),
        requestDevice: async descriptor => { requested = descriptor; return device; } }),
} } });
const host = await openWebGPU();
assert.equal(requested.requiredLimits.maxBufferSize, limits.maxBufferSize);
assert.equal(requested.requiredLimits.maxTextureDimension2D, limits.maxTextureDimension2D);
assert.equal(host.maxBufferSize, limits.maxBufferSize);
assert.equal(host.maxTextureDimension2D, limits.maxTextureDimension2D);
host.close();
const absentLimits = new Proxy({}, { get: (_, key) =>
    ["maxBufferSize", "maxTextureDimension2D"].includes(key) ? undefined : 256 });
navigator.gpu.requestAdapter = async () => ({ info: { isFallbackAdapter: false }, limits: absentLimits,
    features: new Set(), requestDevice: async descriptor => { requested = descriptor; return device; } });
const fallbackHost = await openWebGPU();
assert.equal(Object.hasOwn(requested.requiredLimits, "maxBufferSize"), false);
assert.equal(Object.hasOwn(requested.requiredLimits, "maxTextureDimension2D"), false);
assert.equal(fallbackHost.maxBufferSize, 256);
assert.equal(fallbackHost.maxTextureDimension2D, 256);
fallbackHost.close();
