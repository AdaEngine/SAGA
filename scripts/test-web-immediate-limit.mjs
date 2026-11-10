import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
const limits = new Proxy({ maxImmediateSize: 128 }, { get: (target, key) => target[key] ?? 256 });
let requested;
const device = {
    limits: new Proxy({}, { get: (_, key) => key === "maxImmediateSize" ? (requested.requiredLimits.maxImmediateSize ?? 64) : 256 }),
    features: new Set(), lost: new Promise(() => {}),
    pushErrorScope() {}, popErrorScope: async () => null, createTexture: () => ({ destroy() {} }),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}), destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: {
    wgslLanguageFeatures: new Set(["immediate_address_space"]), requestAdapter: async () => ({
        info: { isFallbackAdapter: false }, limits, features: new Set(),
        requestDevice: async descriptor => { requested = descriptor; return device; },
    }),
} } });
const host = await openWebGPU();
assert.equal(requested.requiredLimits.maxImmediateSize, 128);
assert.equal(host.maxImmediateSize, 128);
host.close();
navigator.gpu.wgslLanguageFeatures.delete("immediate_address_space");
const withoutFeature = await openWebGPU();
assert.equal(requested.requiredLimits.maxImmediateSize, undefined);
assert.equal(withoutFeature.maxImmediateSize, 0);
withoutFeature.close();
navigator.gpu.wgslLanguageFeatures.add("immediate_address_space");
delete limits.maxImmediateSize;
// Old implementations may expose the language feature without the limit property.
// Use a separate adapter object whose absent limit does not inherit the proxy fallback.
const oldLimits = new Proxy({}, { get: (_, key) => key === "maxImmediateSize" ? undefined : 256 });
navigator.gpu.requestAdapter = async () => ({
    info: { isFallbackAdapter: false }, limits: oldLimits, features: new Set(),
    requestDevice: async descriptor => { requested = descriptor; return device; },
});
const oldHost = await openWebGPU();
assert.equal(requested.requiredLimits.maxImmediateSize, undefined);
oldHost.close();
