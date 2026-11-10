import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
globalThis.GPUMapMode = { READ: 1 };
globalThis.VideoFrame = class { codedWidth = 1; close() { this.codedWidth = 0; } };
const limits = new Proxy({}, { get: () => 256 });
let gate, finishMap, passes = 0;
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    createComputePipelineAsync: async () => ({}), createPipelineLayout: () => ({}),
    importExternalTexture: () => ({}), createTexture: () => ({ destroy() {} }),
    createBuffer: () => ({ mapAsync: () => new Promise(resolve => { finishMap = resolve; }),
        getMappedRange: () => new ArrayBuffer(4), unmap() {}, destroy() {} }),
    pushErrorScope() {}, popErrorScope: () => gate ?? Promise.resolve(null),
    createCommandEncoder: () => ({ beginComputePass() { passes++; return {}; }, finish: () => ({}) }),
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
const frame = new VideoFrame(), token = host.registerVideoFrame(frame);
await host.handle(new Packet().word(26).id(2).wide(Number(token)).word(0).data());
await host.handle(new Packet().word(3).id(4).word(0).word(0).word(0).word(0).data());
let release;
gate = new Promise(resolve => { release = () => resolve(null); });
const blocker = host.handle(new Packet().word(8).id(3).word(1).word(1).word(1).word(1)
    .word(1).word(1).word(1).word(0).data());
await Promise.resolve();
const pending = host.handle(new Packet().word(6).word(1).word(2).id(4).word(1).word(1).word(1)
    .word(0).word(1).word(0).word(0).id(2).word(0).word(0).data());
frame.close();
release();
await assert.rejects(pending, error => error.status === 1 && /VideoFrame/.test(error.message));
await blocker;
assert.equal(passes, 0);
host.close();

// Recheck the frame after a readback await inside the submission as well.
gate = undefined;
const second = await openWebGPU();
const nextFrame = new VideoFrame(), nextToken = second.registerVideoFrame(nextFrame);
await second.handle(new Packet().word(26).id(2).wide(Number(nextToken)).word(0).data());
await second.handle(new Packet().word(3).id(4).word(0).word(0).word(0).word(0).data());
await second.handle(new Packet().word(1).id(5).wide(4).word(0).word(0).data());
const afterRead = second.handle(new Packet().word(6).word(2).word(4).id(5).wide(4)
    .word(2).id(4).word(1).word(1).word(1).word(0).word(1).word(0).word(0).id(2).word(0).word(0).data());
while (!finishMap) await Promise.resolve();
nextFrame.close();
finishMap();
await assert.rejects(afterRead, error => error.status === 1 && /VideoFrame/.test(error.message));
assert.equal(passes, 0);
second.close();
