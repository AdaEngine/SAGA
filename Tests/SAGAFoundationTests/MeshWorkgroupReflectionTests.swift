import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskWorkgroupStorageReflectionCountsReachableGlobalsSeparatelyFromPayload() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> a: u32;
    var<workgroup> b: array<u32, 5>;
    var<workgroup> unused: array<u32, 100>;
    fn use_shared() { a = b[0]; }
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u {
        use_shared(); payload.value = a; return vec3u(1);
    }
    """
    let stage = try IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl)
    #expect(stage.workgroupStorageByteSize == 48)
    #expect(stage.payload?.size == 4)
}

@Test func meshWorkgroupStorageReflectionIncludesOutputAndReachableSharedMemory() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vc: u32, @builtin(primitive_count) pc: u32, }
    var<workgroup> output: O;
    var<workgroup> shared: u32;
    var<workgroup> unused: array<u32, 100>;
    @mesh(output) @workgroup_size(1) fn main() { shared = 1; }
    """
    let stage = try IndependentShaderCompiler().compileReflectedMeshStage(.wgsl(source), entryPoint: "main", to: .msl)
    #expect(stage.workgroupStorageByteSize == 96)
}
