import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
const limits = new Proxy({}, { get: () => 256 });
let sampler;
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: () => ({ destroy() {} }),
    createSampler: descriptor => { sampler = descriptor; return {}; },
    destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { wgslLanguageFeatures: new Set(), requestAdapter: async () => ({
    info: { isFallbackAdapter: false }, limits, features: new Set(), requestDevice: async () => device,
}) } } });
const host = await openWebGPU();
const packet = new Uint8Array(68);
const view = new DataView(packet.buffer);
view.setUint32(0, 13, true);
view.setBigUint64(4, 1n, true);
view.setBigUint64(12, 1n, true);
// U, V, W, min, mag, mip, LOD clamps, compare
[0, 1, 2, 0, 1, 0].forEach((value, index) => view.setUint32(20 + index * 4, value, true));
view.setFloat64(44, 1.25, true);
view.setFloat64(52, 3.5, true);
view.setUint32(60, 0, true);
view.setUint32(64, 1, true);
await host.handle(packet);
assert.equal(sampler.addressModeU, "clamp-to-edge");
assert.equal(sampler.addressModeV, "repeat");
assert.equal(sampler.addressModeW, "mirror-repeat");
assert.equal(sampler.magFilter, "linear");
assert.equal(sampler.lodMinClamp, 1.25);
assert.equal(sampler.lodMaxClamp, 3.5);
assert.equal(sampler.maxAnisotropy, 1);
view.setBigUint64(12, 2n, true);
view.setUint32(64, 2, true);
await assert.rejects(host.handle(packet), /Invalid WebGPU sampler descriptor/);
[1, 1, 1].forEach((value, index) => view.setUint32(32 + index * 4, value, true));
view.setUint32(64, 65535, true);
await host.handle(packet);
assert.equal(sampler.maxAnisotropy, 65535, "the browser normalizes values above its limit");
view.setBigUint64(12, 3n, true);
view.setUint32(64, 0, true);
await assert.rejects(host.handle(packet), /Invalid WebGPU sampler descriptor/);
view.setUint32(64, 65536, true);
await assert.rejects(host.handle(packet), /Invalid WebGPU sampler descriptor/);
host.close();
