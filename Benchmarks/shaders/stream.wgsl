// Memory-bound: out = a * 3 + b (two reads and one write per element).
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * 262144u + id.x;
    if (i < arrayLength(&out)) {
        out[i] = a[i] * 3u + b[i];
    }
}
