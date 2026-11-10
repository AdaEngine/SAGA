import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskGLSLUsesPayloadAndUniformDispatch() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(4)
    fn main(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
        if lane == 0u { payload.value = 9u; return vec3u(2u, 1u, 1u); }
        return vec3u(1u);
    }
    """
    let artifact = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: .glsl)
    guard case .text(_, let text) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("#extension GL_EXT_mesh_shader : require"))
    #expect(text.contains("taskPayloadSharedEXT"))
    #expect(text.contains("saga_payload"))
    #expect(text.contains("saga_dispatch_size = saga_size"))
    #expect(text.contains("EmitMeshTasksEXT(saga_dispatch_size.x, saga_dispatch_size.y, saga_dispatch_size.z)"))
}

@Test func taskAndMeshGLSLUseMatchingPayloadDeclaration() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<Vertex, 3>, @builtin(primitives) primitives: array<Primitive, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> output: Output;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u { payload.value = 1u; return vec3u(1u); }
    @mesh(output) @payload(payload) @workgroup_size(1)
    fn mesh() { output.vertex_count = payload.value; output.primitive_count = 0u; }
    """
    let compiler = IndependentShaderCompiler()
    guard case .text(_, let task) = try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl),
          case .text(_, let mesh) = try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let taskDeclaration = task.split(separator: "\n").first { $0.hasPrefix("taskPayloadSharedEXT ") }
    let meshDeclaration = mesh.split(separator: "\n").first { $0.hasPrefix("taskPayloadSharedEXT ") }
    #expect(taskDeclaration != nil)
    #expect(taskDeclaration == meshDeclaration)
}
