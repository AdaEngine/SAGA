import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslTaskRejectsMissingPayload() {
    let source = """
    enable wgpu_mesh_shader;
    @task @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl)
    }
}

@Test func glslTaskCanUseSampledAndStorageTextures() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var smp: sampler;
    @group(0) @binding(2) var destination: texture_storage_2d<rgba8unorm, write>;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u {
        let value = textureSampleLevel(image, smp, vec2f(0.5), 0.0);
        textureStore(destination, vec2i(0), value);
        payload.value = u32(value.x);
        return vec3u(1u);
    }
    """
    guard case .text(_, let code) = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl) else {
        throw ShaderFailure.incompatibleArtifact
    }
    #expect(code.contains("uniform sampler2D saga_pair_"))
    #expect(code.contains("writeonly uniform image2D saga_texture_"))
    #expect(code.contains("textureLod("))
    #expect(code.contains("imageStore("))
}

@Test func glslTaskAndMeshDeclareFloat16Extension() throws {
    let source = """
    enable wgpu_mesh_shader, f16;
    struct Payload { value: f16, padding: f16, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u { payload.value = 1.0h; return vec3u(1u); }
    @mesh(out) @payload(payload) @workgroup_size(1)
    fn mesh() { out.vertex_count = u32(payload.value); out.primitive_count = 0u; }
    """
    let compiler = IndependentShaderCompiler()
    let outputs = [try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl),
                   try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl)]
    for artifact in outputs {
        guard case .text(_, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(code.contains("#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require"))
    }
    if case .text(_, let task) = outputs[0] { #expect(task.contains("float16_t(0.0)")) }
}
