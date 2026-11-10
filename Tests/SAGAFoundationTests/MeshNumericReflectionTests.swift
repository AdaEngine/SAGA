import SAGACompiler
import SAGAShaderIR
import Testing

@Test func taskMeshNumericReflectionFollowsReachableHelpers() throws {
    let source = """
    enable wgpu_mesh_shader, f16, wgpu_int16;
    struct Payload { value: u32, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    fn taskValue(value: u32) -> u32 { return u32(f16(value)); }
    fn meshValue(value: u32) -> u32 { return u32(u16(value)); }
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u { payload.value = taskValue(1u); return vec3u(1u); }
    @mesh(out) @payload(payload) @workgroup_size(1)
    fn mesh() { out.vertex_count = meshValue(payload.value); out.primitive_count = 0u; }
    """
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(.wgsl(source), entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(.wgsl(source), entryPoint: "mesh", to: .msl)
    #expect(task.requiresShaderF16)
    #expect(!task.requiresShaderInt16)
    #expect(!task.requiresShaderInt64)
    #expect(mesh.requiresShaderInt16)
    #expect(!mesh.requiresShaderF16)
    #expect(!mesh.requiresShaderInt64)
    #expect(!task.requiresShaderInt64AtomicMinMax && !task.requiresShaderInt64AtomicAllOps)
    #expect(!mesh.requiresShaderInt64AtomicMinMax && !mesh.requiresShaderInt64AtomicAllOps)
}
