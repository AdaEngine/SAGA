import assert from "node:assert/strict";
import { indirectDrawHasCapacity } from "../web/webgpu.mjs";

// Indexed fetches are robust even when no vertex fits; only index and instance
// bounds restrict the indirect arguments. Non-indexed draws do need vertices.
assert.equal(indirectDrawHasCapacity({ indexed: true, vertices: 0, instances: 1, count: 1 }), true);
assert.equal(indirectDrawHasCapacity({ indexed: false, vertices: 0, instances: 1, count: 1 }), false);
assert.equal(indirectDrawHasCapacity({ indexed: false, vertices: 1, instances: 1, count: 1 }), true);
assert.equal(indirectDrawHasCapacity({ indexed: true, vertices: 1, instances: 1, count: 1 }), true);
assert.equal(indirectDrawHasCapacity({ indexed: true, vertices: 1, instances: 0, count: 1 }), false);
assert.equal(indirectDrawHasCapacity({ indexed: true, vertices: 1, instances: 1, count: 0 }), false);
