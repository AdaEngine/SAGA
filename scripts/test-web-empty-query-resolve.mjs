import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
const limits = new Proxy({}, { get: () => 256 });
const calls = [];
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: () => ({ destroy() {} }),
    createBuffer: () => ({ destroy() {} }), createQuerySet: () => ({ destroy() {} }),
    createCommandEncoder: () => ({
        resolveQuerySet: (...args) => calls.push(["resolve", ...args]),
        finish: () => ({}),
    }),
    queue: { submit: () => calls.push(["submit"]), onSubmittedWorkDone: async () => {} }, destroy() {},
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
await host.handle(new Packet().word(1).id(1).wide(256).word(0).word(0).data());
await host.handle(new Packet().word(17).id(2).word(1).word(1).word(0).data());
const resolve = (first, count) => new Packet().word(6).word(1).word(13)
    .id(2).word(first).word(count).id(1).wide(0).data();
await host.handle(resolve(1, 0));
assert.deepEqual(calls, [], "empty resolve at the query set boundary must not encode or submit");
await host.handle(resolve(0, 1));
assert.equal(calls.length, 2);
assert.equal(calls[0][0], "resolve");
assert.equal(calls[0][2], 0);
assert.equal(calls[0][3], 1);
assert.equal(calls[1][0], "submit");
host.close();
