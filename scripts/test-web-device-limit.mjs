import assert from "node:assert/strict";
import { sampledTextureLimitFromDevice } from "../web/webgpu.mjs";
for (const bad of [undefined, null, 0, -1, 1.5, NaN, Infinity, 4294967296, "32"]) {
    assert.throws(() => sampledTextureLimitFromDevice({ limits: { maxSampledTexturesPerShaderStage: bad } }));
}
assert.throws(() => sampledTextureLimitFromDevice({}));
for (const limit of [16, 32, 4294967295]) {
    assert.equal(sampledTextureLimitFromDevice({ limits: { maxSampledTexturesPerShaderStage: limit } }), limit);
}
console.log("WebGPU device limit validation passed");
