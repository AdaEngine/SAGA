import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
globalThis.GPUMapMode = { READ: 1 };
const limits = new Proxy({}, { get: () => 256 });
let finishMap, clears = 0;
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: () => ({ destroy() {} }),
    createBuffer: () => ({
        mapAsync: () => new Promise(resolve => { finishMap = resolve; }),
        getMappedRange: () => new ArrayBuffer(4), unmap() {}, destroy() {},
    }),
    createCommandEncoder: () => ({ clearBuffer() { clears++; }, finish: () => ({}) }),
    queue: { submit() {}, onSubmittedWorkDone: async () => {} }, destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: {
    wgslLanguageFeatures: new Set(), requestAdapter: async () => ({
        info: { isFallbackAdapter: false }, limits, features: new Set(), requestDevice: async () => device,
    }),
} } });
class Packet {
    bytes = [];
    word(value) { for (let i = 0; i < 4; i++) this.bytes.push((value >>> (8 * i)) & 255); return this; }
    wide(value) { return this.word(value).word(0); }
    id(value) { return this.wide(1).wide(value); }
    data() { return Uint8Array.from(this.bytes); }
}
const host = await openWebGPU();
for (const id of [1, 2]) await host.handle(new Packet().word(1).id(id).wide(4).word(0).word(0).data());
const submission = host.handle(new Packet().word(6).word(2)
    .word(4).id(1).wide(4)
    .word(16).id(2).wide(0).word(1).wide(4).data());
while (!finishMap) await Promise.resolve();
await host.handle(new Packet().word(2).id(2).data());
finishMap();
await assert.rejects(submission, error => error.status === 1 && /resource.*released/i.test(error.message));
assert.equal(clears, 0);
host.close();
