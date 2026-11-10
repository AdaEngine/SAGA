import assert from "node:assert/strict";
import { openWebGPU } from "../web/webgpu.mjs";

class ValidationError extends Error {}
globalThis.GPUValidationError = ValidationError;
globalThis.GPUOutOfMemoryError = class extends Error {};
globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 };
globalThis.GPUBufferUsage = { INDIRECT: 256, COPY_SRC: 4 };
const limits = new Proxy({}, { get: () => 256 });
const scopes = [];
const device = {
    limits, features: new Set(), lost: new Promise(() => {}),
    createShaderModule: () => ({}), createRenderPipelineAsync: async () => ({}),
    pushErrorScope() { scopes.push(null); },
    async popErrorScope() { return scopes.pop(); },
    createTexture: () => ({ destroy() {} }),
    createBuffer: ({ size }) => {
        if (size === 8) scopes[scopes.length - 1] = new ValidationError("second request");
        return { size, destroy() {} };
    },
    destroy() {},
};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: {
    wgslLanguageFeatures: new Set(), requestAdapter: async () => ({
        info: { isFallbackAdapter: false }, limits, features: new Set(), requestDevice: async () => device,
    }),
} } });
const host = await openWebGPU();
function packet(id, size) {
    const bytes = new Uint8Array(36);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 1, true);
    view.setBigUint64(4, 1n, true);
    view.setBigUint64(12, BigInt(id), true);
    view.setBigUint64(20, BigInt(size), true);
    return bytes;
}
const results = await Promise.allSettled([host.handle(packet(1, 4)), host.handle(packet(2, 8))]);
assert.equal(results[0].status, "fulfilled");
assert.equal(results[1].status, "rejected");
assert.equal(results[1].reason.message, "second request");
assert.equal(results[1].reason.status, 1);
assert.equal(host.metrics().remainingResources, 1);
host.close();
