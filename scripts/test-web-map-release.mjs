import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
const limits = new Proxy({}, { get: () => 256 });
let finishMap;
let buffer;
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: () => ({ destroy() {} }),
    createBuffer: () => (buffer = {
        mapState: "unmapped", destroyed: false,
        mapAsync() { return new Promise(resolve => { finishMap = () => { this.mapState = "mapped"; resolve(); }; }); },
        getMappedRange: () => new ArrayBuffer(4),
        unmap() { this.mapState = "unmapped"; },
        destroy() { this.destroyed = true; },
    }),
    destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { wgslLanguageFeatures: new Set(), requestAdapter: async () => ({
    info: { isFallbackAdapter: false }, limits, features: new Set(), requestDevice: async () => device,
}) } } });
const packet = (operation, size) => {
    const bytes = new Uint8Array(size);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, operation, true);
    view.setBigUint64(4, 1n, true);
    view.setBigUint64(12, 1n, true);
    return { bytes, view };
};
const host = await openWebGPU();
const create = packet(1, 36);
create.view.setBigUint64(20, 4n, true);
await host.handle(create.bytes);
const mapping = packet(19, 40);
mapping.view.setUint32(20, GPUMapMode.READ, true);
mapping.view.setBigUint64(32, 4n, true);
const pending = host.handle(mapping.bytes);
while (!finishMap) await Promise.resolve();
await host.handle(packet(2, 20).bytes);
finishMap();
await assert.rejects(pending);
assert.equal(buffer.mapState, "unmapped");
assert.equal(host.metrics().remainingResources, 0);
host.close();
