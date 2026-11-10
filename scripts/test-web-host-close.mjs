import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
const limits = new Proxy({}, { get: () => 256 });
let destroyedBuffers = 0;
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: () => ({ destroy() {} }),
    createBuffer: ({ size }) => ({ size, destroy() { destroyedBuffers += 1; } }),
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
await host.handle(packet);
assert.equal(host.metrics().remainingResources, 1);
host.close();
assert.equal(host.metrics().remainingResources, 0);
await assert.rejects(host.handle(packet), { status: 3 });

// A request already inside an asynchronous error scope must not repopulate the host after close.
const closingHost = await openWebGPU();
const pending = closingHost.handle(packet);
await Promise.resolve(); // Allow createBuffer to run before closing.
closingHost.close();
await assert.rejects(pending, { status: 3 });
assert.equal(closingHost.metrics().remainingResources, 0);
assert.equal(destroyedBuffers, 1, "closing during allocation must destroy the newly created buffer");

// Exercise close at successive microtask boundaries, including after the error
// scope resolves but before the handle continuation can publish the buffer.
for (let delay = 0; delay < 14; delay += 1) {
    const racingHost = await openWebGPU();
    const before = destroyedBuffers;
    const allocation = racingHost.handle(packet);
    let remaining = delay;
    const closeLater = () => remaining-- > 0 ? queueMicrotask(closeLater) : racingHost.close();
    queueMicrotask(closeLater);
    const outcome = await Promise.allSettled([allocation]);
    while (remaining >= 0) await Promise.resolve();
    assert.equal(racingHost.metrics().remainingResources, 0);
    if (outcome[0].status === "rejected" && delay >= 1) {
        assert.equal(destroyedBuffers, before + 1, `allocation cleanup at microtask ${delay}`);
    }
}

// A queued presentation must not use a texture released before encoding.
let finishWait;
let copies = 0;
device.queue = {
    onSubmittedWorkDone: () => finishWait ? Promise.resolve() : new Promise(resolve => { finishWait = resolve; }),
    submit() {},
};
device.createTexture = descriptor => ({ ...descriptor, width: 1, height: 1, format: descriptor.format, destroy() {} });
device.createCommandEncoder = () => ({ copyTextureToTexture() { copies += 1; }, finish: () => ({}) });
const context = { configure() {}, unconfigure() {}, getCurrentTexture: () => ({}) };
const canvas = { width: 1, height: 1, getContext: () => context };
const surfaceHost = await openWebGPU(canvas);
function surfacePacket(op) {
    const bytes = new Uint8Array(op === 8 ? 52 : op === 7 ? 4 : 20);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, op, true);
    if (op !== 7) { view.setBigUint64(4, 1n, true); view.setBigUint64(12, 1n, true); }
    if (op === 8) [1, 1, 1, 1, 1, 1, 1, 0].forEach((v, i) => view.setUint32(20 + i * 4, v, true));
    return bytes;
}
await surfaceHost.handle(surfacePacket(8));
const wait = surfaceHost.handle(surfacePacket(7));
while (!finishWait) await Promise.resolve();
const presentation = surfaceHost.handle(surfacePacket(12));
await surfaceHost.handle(surfacePacket(9));
finishWait();
await wait;
await assert.rejects(presentation, /released/);
assert.equal(copies, 0);
surfaceHost.close();

// A queued submission must not encode a buffer that was released while it waited.
finishWait = undefined;
let clears = 0;
device.createCommandEncoder = () => ({ clearBuffer() { clears += 1; }, finish: () => ({}) });
const submissionHost = await openWebGPU();
await submissionHost.handle(packet);
const blockingWait = submissionHost.handle(surfacePacket(7));
while (!finishWait) await Promise.resolve();
const command = new Uint8Array(56);
const commandView = new DataView(command.buffer);
commandView.setUint32(0, 6, true);
commandView.setUint32(4, 1, true);
commandView.setUint32(8, 16, true);
commandView.setBigUint64(12, 1n, true);
commandView.setBigUint64(20, 1n, true);
commandView.setBigUint64(28, 0n, true);
commandView.setUint32(36, 1, true);
commandView.setBigUint64(40, 4n, true);
const queued = submissionHost.handle(command.subarray(0, 48));
await submissionHost.handle(surfacePacket(2));
finishWait();
await blockingWait;
await assert.rejects(queued, /released/);
assert.equal(clears, 0);
submissionHost.close();
