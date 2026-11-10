import Testing
import SAGACompiler
import SAGAShaderIR

@Test func metalTaskWorkgroupMemoryAndPayloadBarrier() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> shared: array<u32, 8>;
    @task @payload(payload) @workgroup_size(4)
    fn main(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
        shared[lane] = lane;
        payload.value = shared[lane];
        workgroupBarrier();
        return vec3u(1);
    }
    """
    let stage = try IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl)
    guard case .text(_, let text) = stage.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("threadgroup uint saga_shared_"))
    #expect(text.contains("mem_flags::mem_object_data"))
}

@Test(arguments: ["task", "mesh"])
func metalMeshStagesRejectOverflowingWorkgroupProduct(stage: String) {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(16384, 16384, 16384)
        fn task() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vc: u32, @builtin(primitive_count) pc: u32, }
        var<workgroup> output: O;
        @mesh(output) @workgroup_size(16384, 16384, 16384) fn mesh() {}
        """)
    #expect(throws: ShaderFailure.self) {
        let compiler = IndependentShaderCompiler()
        if stage == "task" { _ = try compiler.compileReflectedTaskStage(source, entryPoint: stage, to: .msl) }
        else { _ = try compiler.compileReflectedMeshStage(source, entryPoint: stage, to: .msl) }
    }
}
