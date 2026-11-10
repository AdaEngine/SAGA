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
    createBuffer: () => { const buffer = { destroyed: false, destroy() { this.destroyed = true; } }; created.push(buffer); return buffer; },
    destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { wgslLanguageFeatures: new Set(), requestAdapter: async () => ({
    info: { isFallbackAdapter: false }, limits, features: new Set(), requestDevice: async () => device,
}) } } });
const host = await openWebGPU();
function packet(operation, id = 1n) {
    const bytes = new Uint8Array(operation === 1 ? 36 : operation === 5 ? 12 : 20);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, operation, true);
    view.setBigUint64(4, 1n, true);
    if (operation !== 5) view.setBigUint64(12, id, true);
    if (operation === 1) view.setBigUint64(20, 4n, true);
    return bytes;
}
for (const release of [2, 5]) {
    const allocation = host.handle(packet(1));
    await host.handle(packet(release));
    await assert.rejects(allocation, /released during allocation/);
    assert.equal(host.metrics().remainingResources, 0);
    assert.equal(created.at(-1).destroyed, true);
    // A later allocation of the same ID remains valid.
    await host.handle(packet(1));
    assert.equal(host.metrics().remainingResources, 1);
    await host.handle(packet(2));
}
host.close();
