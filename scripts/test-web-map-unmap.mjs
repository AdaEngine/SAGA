import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
const limits = new Proxy({}, { get: () => 256 });
let finishMap;
let rejectMap;
let buffer;
let finishWork;
let mapCalls = 0;
const device = {
    queue: { onSubmittedWorkDone: () => new Promise(resolve => { finishWork = resolve; }) },
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: () => ({ destroy() {} }),
    createBuffer: () => (buffer = {
        mapState: "unmapped", destroyed: false,
        mapAsync() { mapCalls++; this.mapState = "pending"; return new Promise((resolve, reject) => { rejectMap = reject; finishMap = () => { this.mapState = "mapped"; resolve(); }; }); },
        getMappedRange: () => new ArrayBuffer(4),
        unmap() { this.mapState = "unmapped"; rejectMap?.(new Error("mapping cancelled")); },
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
await host.handle(packet(22, 20).bytes);
if (buffer.mapState === "pending") finishMap();
await assert.rejects(pending);
assert.equal(buffer.mapState, "unmapped");
assert.equal(host.metrics().remainingResources, 1);
// A map waiting for another operation's error scopes is cancelled too.
const waitPacket = new Uint8Array(4);
new DataView(waitPacket.buffer).setUint32(0, 7, true);
const work = host.handle(waitPacket);
while (!finishWork) await Promise.resolve();
const queued = host.handle(mapping.bytes);
await host.handle(packet(22, 20).bytes);
finishWork();
await work;
await assert.rejects(queued);
assert.equal(mapCalls, 1);
assert.equal(buffer.mapState, "unmapped");
assert.equal(host.metrics().remainingResources, 1);
// Cancelling one request must not unmap a subsequent request on the same buffer.
finishMap = undefined;
const oldMap = host.handle(mapping.bytes);
while (!finishMap) await Promise.resolve();
await host.handle(packet(22, 20).bytes);
finishMap = undefined;
const newMap = host.handle(mapping.bytes);
await assert.rejects(oldMap);
while (!finishMap) await Promise.resolve();
finishMap();
await newMap;
assert.equal(buffer.mapState, "mapped");
await host.handle(packet(22, 20).bytes);
// Releasing an ID clears its pending state before that ID can be reused.
finishMap = undefined;
const releasedMap = host.handle(mapping.bytes);
while (!finishMap) await Promise.resolve();
await host.handle(packet(2, 20).bytes);
const recreate = host.handle(create.bytes);
finishMap();
await assert.rejects(releasedMap);
await recreate;
finishMap = undefined;
const recreatedMap = host.handle(mapping.bytes);
while (!finishMap) await Promise.resolve();
finishMap();
await recreatedMap;
assert.equal(buffer.mapState, "mapped");
await host.handle(packet(22, 20).bytes);
host.close();
