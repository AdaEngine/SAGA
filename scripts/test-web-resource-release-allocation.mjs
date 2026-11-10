import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
const limits = new Proxy({}, { get: () => 256 });
const created = [];
const resource = kind => { const value = { kind, destroyed: false, destroy() { this.destroyed = true; } }; created.push(value); return value; };
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture: () => Object.assign(resource("texture"), { createView: () => resource("view") }), createQuerySet: () => resource("query"),
    createSampler: () => resource("sampler"), createComputePipelineAsync: async () => resource("compute"),
    createPipelineLayout: () => ({}), importExternalTexture: () => resource("external"), destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { wgslLanguageFeatures: new Set(), requestAdapter: async () => ({
    info: { isFallbackAdapter: false }, limits, features: new Set(), requestDevice: async () => device,
}) } } });
function packet(operation, id = 1n) {
    const length = ({ 8: 52, 13: 68, 17: 32, 5: 12 })[operation] ?? 20;
    const bytes = new Uint8Array(length), view = new DataView(bytes.buffer);
    view.setUint32(0, operation, true); view.setBigUint64(4, 1n, true);
    if (operation !== 5) view.setBigUint64(12, id, true);
    if (operation === 8) [1, 1, 1, 1, 1, 1, 1, 0].forEach((v, i) => view.setUint32(20 + i * 4, v, true));
    if (operation === 13) { view.setFloat64(52, 32, true); view.setUint32(64, 1, true); }
    if (operation === 17) view.setUint32(24, 1, true);
    return bytes;
}
for (const [create, release, kind] of [[8, 9, "texture"], [13, 14, "sampler"], [17, 18, "query"]]) {
    for (const releaseOperation of [release, 5]) {
        const host = await openWebGPU();
        const allocation = host.handle(packet(create));
        await host.handle(packet(releaseOperation));
        const replacement = host.handle(packet(create));
        await assert.rejects(allocation, /released during allocation/);
        await replacement;
        assert.equal(host.metrics().remainingResources, 1, `${kind} replacement remains live`);
        if (kind !== "sampler") assert.equal(created.at(-2).destroyed, true);
        await host.handle(packet(release));
        assert.equal(host.metrics().remainingResources, 0);
        host.close();
    }
    const host = await openWebGPU();
    const allocation = host.handle(packet(create));
    host.close();
    await assert.rejects(allocation, /closed/);
    assert.equal(host.metrics().remainingResources, 0);
}

// Construct packets with empty lists and strings for the remaining resource types.
function words(operation, values, id = 2n) {
    const bytes = new Uint8Array(20 + values.length * 4), view = new DataView(bytes.buffer);
    view.setUint32(0, operation, true); view.setBigUint64(4, 1n, true); view.setBigUint64(12, id, true);
    values.forEach((value, i) => view.setUint32(20 + i * 4, value, true));
    return bytes;
}
globalThis.VideoFrame = class { constructor() { this.codedWidth = 1; } };
for (const [create, release] of [[3, 4], [10, 11], [15, 16], [26, 27]]) {
    for (const releaseOperation of [release, 5]) {
        const host = await openWebGPU();
        let input;
        if (create === 3) input = words(3, [0, 0, 0, 0]);
        if (create === 10) input = words(10, [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        if (create === 15) {
            await host.handle(packet(8));
            input = words(15, [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
        }
        if (create === 26) {
            const token = host.registerVideoFrame(new VideoFrame());
            input = words(26, [Number(token), 0, 0]);
        }
        const allocation = host.handle(input);
        await host.handle(packet(releaseOperation, 2n));
        if (create === 15 && releaseOperation === 5) await host.handle(packet(8));
        const replacement = host.handle(input);
        await assert.rejects(allocation, /released during allocation/);
        await replacement;
        await host.handle(packet(release, 2n));
        host.close();
    }
}
// Closing while error scopes are being popped must discard an already-created object.
for (const [create, kind] of [[8, "texture"], [13, "sampler"], [17, "query"]]) {
    const host = await openWebGPU();
    let releaseScopes;
    const gate = new Promise(resolve => { releaseScopes = resolve; });
    device.popErrorScope = () => gate.then(() => null);
    const before = created.length;
    const allocation = host.handle(packet(create));
    while (created.length === before) await Promise.resolve();
    const object = created.at(-1);
    assert.equal(object.kind, kind);
    host.close();
    releaseScopes();
    await assert.rejects(allocation, /closed/);
    assert.equal(object.destroyed, true);
    device.popErrorScope = async () => null;
}

// A video frame registration must stay live throughout external texture import.
for (const phase of ["queued", "scopes", "closed-frame"]) {
    const host = await openWebGPU();
    const frame = new VideoFrame();
    const token = host.registerVideoFrame(frame);
    let releaseScopes;
    const gate = new Promise(resolve => { releaseScopes = resolve; });
    let imports = 0;
    device.importExternalTexture = () => { imports++; return resource("external"); };
    if (phase === "queued") {
        device.popErrorScope = () => gate.then(() => null);
        const blocker = host.handle(packet(13));
        while (created.at(-1)?.kind !== "sampler") await Promise.resolve();
        const allocation = host.handle(words(26, [Number(token), 0, 0]));
        host.unregisterVideoFrame(token);
        releaseScopes();
        await blocker;
        await assert.rejects(allocation, /VideoFrame/);
        assert.equal(imports, 0);
    } else {
        device.popErrorScope = () => gate.then(() => null);
        const allocation = host.handle(words(26, [Number(token), 0, 0]));
        while (!imports) await Promise.resolve();
        if (phase === "scopes") host.unregisterVideoFrame(token);
        else frame.codedWidth = 0;
        releaseScopes();
        await assert.rejects(allocation, /VideoFrame/);
    }
    assert.equal(host.metrics().remainingResources, phase === "queued" ? 1 : phase === "closed-frame" ? 1 : 0);
    host.close();
    device.popErrorScope = async () => null;
}
