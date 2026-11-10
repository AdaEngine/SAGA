import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

const names = ["maxTextureDimension1D", "maxTextureDimension3D", "maxTextureArrayLayers",
    "maxBindGroups", "maxBindingsPerBindGroup", "maxDynamicUniformBuffersPerPipelineLayout",
    "maxDynamicStorageBuffersPerPipelineLayout", "maxSampledTexturesPerShaderStage",
    "maxSamplersPerShaderStage", "maxStorageBuffersPerShaderStage", "maxStorageTexturesPerShaderStage",
    "maxUniformBuffersPerShaderStage", "maxUniformBufferBindingSize", "maxStorageBufferBindingSize"];
const limits = new Proxy(Object.fromEntries(names.map((name, i) => [name, 1024 + i])),
    { get: (target, key) => target[key] ?? 256 });
let requested;
const device = {
    limits: new Proxy({}, { get: (_, key) => requested.requiredLimits[key] ?? 256 }),
    features: new Set(), lost: new Promise(() => {}),
    pushErrorScope() {}, popErrorScope: async () => null, createTexture: () => ({ destroy() {} }),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}), destroy() {},
};
globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: {
    requestAdapter: async () => ({ info: { isFallbackAdapter: false }, limits, features: new Set(),
        requestDevice: async descriptor => { requested = descriptor; return device; } }),
} } });
const host = await openWebGPU();
for (const name of names) assert.equal(requested.requiredLimits[name], limits[name], name);
assert.equal(requested.requiredLimits.maxBindGroupsPlusVertexBuffers, 256);
assert.equal(host.maxBindGroupsPlusVertexBuffers, 256);
host.close();
const absentLimits = new Proxy({}, { get: (_, key) => names.includes(key) ? undefined : 256 });
navigator.gpu.requestAdapter = async () => ({ info: { isFallbackAdapter: false }, limits: absentLimits,
    features: new Set(), requestDevice: async descriptor => { requested = descriptor; return device; } });
const fallbackHost = await openWebGPU();
for (const name of names) assert.equal(Object.hasOwn(requested.requiredLimits, name), false, name);
fallbackHost.close();

// Older implementations may not expose the combined limit. Use the WebGPU
// default rather than passing an undefined value through the Wasm ABI.
const noCombined = new Proxy({}, { get: (_, key) => key === "maxBindGroupsPlusVertexBuffers" ? undefined : 256 });
navigator.gpu.requestAdapter = async () => ({ info: { isFallbackAdapter: false }, limits: noCombined,
    features: new Set(), requestDevice: async descriptor => {
        requested = descriptor;
        return { ...device, limits: noCombined };
    } });
const legacyHost = await openWebGPU();
assert.equal(Object.hasOwn(requested.requiredLimits, "maxBindGroupsPlusVertexBuffers"), false);
assert.equal(legacyHost.maxBindGroupsPlusVertexBuffers, 24);
legacyHost.close();
