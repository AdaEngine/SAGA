import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, RENDER_ATTACHMENT: 16 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
const limits = new Proxy({}, { get: () => 256 });
const textures = [], copies = [];
let complete;
const retired = new Promise(resolve => { complete = resolve; });
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: descriptor => {
        const texture = { ...descriptor, width: descriptor.size[0], height: descriptor.size[1],
            depthOrArrayLayers: descriptor.size[2] ?? 1, destroyed: false,
            destroy() { this.destroyed = true; } };
        textures.push(texture); return texture;
    },
    createCommandEncoder: () => ({ copyTextureToTexture: (...args) => copies.push(args), finish: () => ({}) }),
    queue: { submit() {}, onSubmittedWorkDone: () => retired }, destroy() {},
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
await host.handle(new Packet().word(8).id(1).word(8).word(8).word(1).word(3).word(2).word(1).word(8).word(1).data());
const original = textures.at(-1);
const command = (srcMip, dstMip, width = 4, id = 1) => new Packet().word(6).word(1).word(9)
    .id(id).id(id).word(0).word(0).word(0).word(srcMip)
    .word(0).word(0).word(2).word(dstMip).word(width).word(4).word(2).data();
const submission = host.handle(command(1, 1));
await new Promise(resolve => setTimeout(resolve, 0));
assert.equal(copies.length, 2, "same 3D mip must copy through a temporary texture");
const staging = textures.at(-1);
assert.notEqual(staging, original);
assert.equal(staging.dimension, "3d");
assert.equal(staging.format, "rgba8unorm");
assert.deepEqual(staging.size, [4, 4, 2]);
assert.equal(staging.usage, GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST);
assert.equal(copies[0][0].texture, original);
assert.equal(copies[0][0].mipLevel, 1);
assert.equal(copies[0][1].texture, staging);
assert.equal(copies[1][0].texture, staging);
assert.equal(copies[1][1].texture, original);
assert.equal(copies[1][1].mipLevel, 1);
assert.deepEqual(copies[1][1].origin, [0, 0, 2]);
assert.equal(staging.destroyed, false, "temporary texture must survive until submitted work retires");
complete();
await submission;
assert.equal(staging.destroyed, true);
const previous = copies.length;
await host.handle(command(0, 1));
assert.equal(copies.length, previous + 1, "different mips need no staging");
await host.handle(command(1, 1, 0));
assert.equal(copies.length, previous + 1, "zero extent must remain a no-op");
await host.handle(new Packet().word(8).id(2).word(8).word(8).word(20).word(3).word(2).word(1).word(8).word(1).data());
await host.handle(command(1, 1, 4, 2));
const compressed = textures.at(-1);
assert.equal(compressed.format, "bc1-rgba-unorm");
assert.deepEqual(compressed.size, [4, 4, 2]);
assert.equal(compressed.destroyed, true);
const before2D = copies.length;
await host.handle(new Packet().word(8).id(3).word(8).word(8).word(1).word(3).word(2).word(1).word(8).word(0).data());
await host.handle(command(1, 1, 4, 3));
assert.equal(copies.length, before2D + 1, "2D array layers do not need staging");
host.close();
