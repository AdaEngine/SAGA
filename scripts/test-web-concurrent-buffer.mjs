import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
const limits = new Proxy({}, { get: () => 256 });
const created = [];
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: () => ({ destroy() {} }),
    createSampler: () => ({}),
    createQuerySet: () => { const resource = { destroyed: false, destroy() { this.destroyed = true; } }; created.push(resource); return resource; },
    createBuffer: () => { const buffer = { destroyed: false, destroy() { this.destroyed = true; } }; created.push(buffer); return buffer; },
    destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { wgslLanguageFeatures: new Set(), requestAdapter: async () => ({
    info: { isFallbackAdapter: false }, limits, features: new Set(), requestDevice: async () => device,
}) } } });
const host = await openWebGPU();
const packet = new Uint8Array(36);
const view = new DataView(packet.buffer);
view.setUint32(0, 1, true);
view.setBigUint64(4, 1n, true);
view.setBigUint64(12, 1n, true);
view.setBigUint64(20, 4n, true);
const outcomes = await Promise.allSettled([host.handle(packet), host.handle(packet)]);
assert.equal(outcomes.filter(outcome => outcome.status === "fulfilled").length, 1);
assert.equal(host.metrics().remainingResources, 1);
assert.equal(created.filter(buffer => !buffer.destroyed).length, 1);
// Query sets have explicit destruction; overlapping allocations must retain one.
const query = new Uint8Array(32);
const queryView = new DataView(query.buffer);
queryView.setUint32(0, 17, true);
queryView.setBigUint64(4, 1n, true);
queryView.setBigUint64(12, 2n, true);
queryView.setUint32(24, 1, true);
const queries = await Promise.allSettled([host.handle(query), host.handle(query)]);
assert.equal(queries.filter(outcome => outcome.status === "fulfilled").length, 1);
assert.equal(created.filter(resource => !resource.destroyed).length, 2);
const sampler = new Uint8Array(68);
const samplerView = new DataView(sampler.buffer);
samplerView.setUint32(0, 13, true);
samplerView.setBigUint64(4, 1n, true);
samplerView.setBigUint64(12, 3n, true);
samplerView.setFloat64(52, 32, true);
samplerView.setUint32(64, 1, true);
const samplers = await Promise.allSettled([host.handle(sampler), host.handle(sampler)]);
assert.equal(samplers.filter(outcome => outcome.status === "fulfilled").length, 1);
assert.equal(host.metrics().remainingResources, 3);
host.close();
