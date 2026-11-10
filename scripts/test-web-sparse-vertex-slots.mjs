import assert from "node:assert/strict";
import { indirectVertexCapacities } from "../web/webgpu.mjs";

const empty = { arrayStride: 32, stepMode: "vertex", attributeEnd: 0 };
const active = { arrayStride: 8, stepMode: "instance", attributeEnd: 8 };
const bindings = [{ slot: 1, buffer: { size: 40 }, offset: 8 }];
assert.deepEqual(indirectVertexCapacities([empty, active], bindings), { vertices: 2 ** 32, instances: 4 });
assert.deepEqual(indirectVertexCapacities([active, empty], [{ slot: 0, buffer: { size: 16 }, offset: 0 }]),
    { vertices: 2 ** 32, instances: 2 });
assert.throws(() => indirectVertexCapacities([empty, active], []), /vertex buffer/i);

// Binding order is independent of pipeline slot order. An additional binding at
// the last legal slot is valid and must not constrain indirect draw capacity.
const vertex = { arrayStride: 4, stepMode: "vertex", attributeEnd: 4 };
assert.deepEqual(indirectVertexCapacities([empty, vertex, empty, active], [
    { slot: 7, buffer: { size: 0 }, offset: 0 },
    { slot: 3, buffer: { size: 40 }, offset: 8 },
    { slot: 1, buffer: { size: 12 }, offset: 0 },
]), { vertices: 3, instances: 4 });

import { webVertexBufferLayout } from "../web/webgpu.mjs";
assert.equal(webVertexBufferLayout(0, "vertex", []), null);
const attribute = { shaderLocation: 0, offset: 0, format: "float32" };
assert.deepEqual(webVertexBufferLayout(4, "vertex", [attribute]),
    { arrayStride: 4, stepMode: "vertex", attributes: [attribute] });
