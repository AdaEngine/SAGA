import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
const limits = new Proxy({}, { get: () => 256 });
let views = 0, scopeGate;
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: () => scopeGate ?? Promise.resolve(null),
    createTexture: () => ({ destroy() {}, createView() { views++; return {}; } }), destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { wgslLanguageFeatures: new Set(), requestAdapter: async () => ({
    info: { isFallbackAdapter: false }, limits, features: new Set(), requestDevice: async () => device,
}) } } });
function packet(op, id, values = []) {
    const bytes = new Uint8Array(20 + values.length * 4), view = new DataView(bytes.buffer);
    view.setUint32(0, op, true); view.setBigUint64(4, 1n, true); view.setBigUint64(12, id, true);
    values.forEach((v, i) => view.setUint32(20 + i * 4, v, true));
    return bytes;
}
const texture = packet(8, 1n, [1, 1, 1, 1, 1, 1, 1, 0]);
const view = packet(15, 2n, [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
for (const afterCreation of [false, true]) {
    const host = await openWebGPU();
    await host.handle(texture);
    let release;
    scopeGate = new Promise(resolve => { release = () => resolve(null); });
    let blocker;
    if (!afterCreation) {
        blocker = host.handle(packet(8, 3n, [1, 1, 1, 1, 1, 1, 1, 0]));
        await Promise.resolve();
    }
    const before = views;
    const allocation = host.handle(view);
    if (afterCreation) while (views === before) await Promise.resolve();
    await host.handle(packet(9, 1n));
    const replacement = host.handle(texture);
    release();
    await assert.rejects(allocation, /texture.*released/i);
    if (blocker) await blocker;
    await replacement;
    assert.equal(host.metrics().remainingResources, blocker ? 2 : 1);
    if (!afterCreation) assert.equal(views, before, "must not create a view of a released texture");
    scopeGate = undefined;
    host.close();
}

// A queued dispatch referring to a view must reject if its parent is destroyed.
device.createComputePipelineAsync = async () => ({});
device.createPipelineLayout = () => ({});
device.queue = { submit() {}, onSubmittedWorkDone: async () => {} };
let passes = 0;
device.createCommandEncoder = () => ({ beginComputePass() { passes++; return {}; }, finish: () => ({}) });
class Packet {
    bytes = [];
    word(value) { for (let i = 0; i < 4; i++) this.bytes.push((value >>> (8 * i)) & 255); return this; }
    wide(value) { return this.word(value).word(0); }
    id(value) { return this.wide(1).wide(value); }
    data() { return Uint8Array.from(this.bytes); }
}
for (const releaseOperation of [9, 16]) {
    const host = await openWebGPU();
    await host.handle(texture);
    await host.handle(view);
    await host.handle(new Packet().word(3).id(4).word(0).word(0).word(0).word(0).data());
    let release;
    scopeGate = new Promise(resolve => { release = () => resolve(null); });
    const before = views;
    const blocker = host.handle(packet(8, 3n, [1, 1, 1, 1, 1, 1, 1, 0]));
    await Promise.resolve();
    const dispatch = new Packet().word(6).word(1).word(2).id(4).word(1).word(1).word(1)
        .word(0).word(1).word(0).word(0).id(2).word(0).word(0).data();
    const pending = host.handle(dispatch);
    await host.handle(packet(releaseOperation, releaseOperation === 9 ? 1n : 2n));
    release();
    await assert.rejects(pending, /released/);
    await blocker;
    assert.equal(passes, 0);
    assert.equal(views, before);
    scopeGate = undefined;
    host.close();
}
