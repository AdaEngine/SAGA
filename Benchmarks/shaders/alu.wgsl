// ALU-bound: every invocation runs a dependent integer hash chain and writes one word.
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * 262144u + id.x; // 1024 workgroups * 256 invocations per row
    if (i < arrayLength(&out)) {
        var x = i;
        for (var k = 0u; k < __ITERS__; k = k + 1u) {
            x = x * 1664525u + 1013904223u;
            x = x ^ (x >> 13u);
        }
        out[i] = x;
    }
}
