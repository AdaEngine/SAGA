import assert from "node:assert/strict";
import { textureBlockRows } from "../web/webgpu.mjs";

assert.equal(textureBlockRows("rgba8unorm", 8), 8);
assert.equal(textureBlockRows("bc1-rgba-unorm", 8), 2);
assert.equal(textureBlockRows("etc2-rgb8unorm", 4), 1);
assert.equal(textureBlockRows("eac-r11unorm", 8), 2);
assert.equal(textureBlockRows("astc-10x8-unorm", 16), 2);

// The regional copy must preserve the origin, extent, and destination image stride.
const { openWebGPU } = await import("../web/webgpu.mjs");
globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
const limits = new Proxy({}, { get: () => 256 });
const copies = [];
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: descriptor => ({ ...descriptor, destroy() {} }),
    createBuffer: descriptor => ({ ...descriptor, destroy() {} }),
    createCommandEncoder: () => ({
        copyTextureToBuffer: (...args) => copies.push(args), finish: () => ({}),
    }),
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
    wide(value) { this.word(value); this.word(0); return this; }
    id(value) { return this.wide(1).wide(value); }
    data() { return Uint8Array.from(this.bytes); }
}
const host = await openWebGPU();
await host.handle(new Packet().word(1).id(1).wide(2048).word(0).word(0).data());
await host.handle(new Packet().word(8).id(2).word(4).word(4).word(1).word(1)
    .word(1).word(1).word(2).word(0).data());
const regional = width => new Packet().word(6).word(1).word(18)
    .id(2).id(1).wide(8).word(1).word(256).word(1).word(4)
    .word(1).word(1).word(0).word(width).word(1).word(2).word(0).word(0).data();
await host.handle(regional(1));
assert.equal(copies.length, 1);
assert.deepEqual(copies[0][0].origin, [1, 1, 0]);
assert.equal(copies[0][0].mipLevel, 0);
assert.equal(copies[0][0].aspect, "all");
assert.deepEqual(copies[0][2], [1, 1, 2]);
assert.equal(copies[0][1].offset, 8);
assert.equal(copies[0][1].bytesPerRow, 256);
assert.equal(copies[0][1].rowsPerImage, 4);
const packed = new Packet().word(6).word(1).word(18)
    .id(2).id(1).wide(8).word(1).word(256).word(0)
    .word(1).word(1).word(0).word(1).word(2).word(2).word(0).word(0).data();
await host.handle(packed);
assert.equal(copies[1][1].rowsPerImage, 2, "packed layered copies need an explicit image stride");
await host.handle(new Packet().word(8).id(3).word(8).word(8).word(20).word(1)
    .word(1).word(1).word(2).word(0).data());
const compressed = new Packet().word(6).word(1).word(18)
    .id(3).id(1).wide(8).word(1).word(256).word(0)
    .word(0).word(0).word(0).word(8).word(8).word(2).word(0).word(0).data();
await host.handle(compressed);
assert.equal(copies[2][1].rowsPerImage, 2, "compressed image stride counts block rows");
await host.handle(regional(0));
assert.equal(copies.length, 3, "empty copies must not reach the browser");
host.close();
