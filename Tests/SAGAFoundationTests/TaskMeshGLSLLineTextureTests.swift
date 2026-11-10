import SAGACompiler
import SAGAShaderIR
import Testing

@Test func taskAndMeshGLSLAcceptNativeLineTextureOffset() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: f32, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    @group(0) @binding(0) var image: texture_1d<f32>;
    @group(0) @binding(1) var state: sampler;
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
      payload.value = textureSampleLevel(image, state, 0.5, 0.0, 1i).x; return vec3u(1u);
    }
    @mesh(out) @workgroup_size(1) fn mesh() {
      let sample = textureSampleLevel(image, state, 0.5, 0.0, 1i);
      out.vertices[0].position = sample; out.vertex_count = 1u; out.primitive_count = 0u;
    }
    """
    let compiler = IndependentShaderCompiler()
    for artifact in [try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl),
                     try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl)] {
        guard case .text(_, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(code.contains("textureLodOffset("))
    }
}
