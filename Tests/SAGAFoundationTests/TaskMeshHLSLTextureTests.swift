import SAGACompiler
import SAGAShaderIR
import Testing

@Test func taskAndMeshHLSLSupportTextureResources() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: f32, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    @group(1) @binding(0) var image: texture_2d<f32>;
    @group(1) @binding(1) var state: sampler;
    @group(1) @binding(2) var target: texture_storage_2d<rgba8unorm, write>;
    var<task_payload> payload: Payload;
    var<workgroup> output: O;
    fn getColor() -> vec4f { return textureSampleLevel(image, state, vec2f(0.5), 0.0); }
    @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
      let value = getColor(); textureStore(target, vec2i(0), value); payload.value = value.x; return vec3u(1u);
    }
    @mesh(output) @workgroup_size(1) fn mesh() {
      let value = getColor(); textureStore(target, vec2i(0), value);
      output.vertices[0].position = value; output.vertex_count = 1u; output.primitive_count = 0u;
    }
    """
    let compiler = IndependentShaderCompiler()
    let artifacts = [try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .hlsl),
                     try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .hlsl)]
    for artifact in artifacts {
        guard case .text(.hlsl, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(code.contains("Texture2D<float4>"))
        #expect(code.contains("SamplerState"))
        #expect(code.contains("RWTexture2D"))
        #expect(code.contains("SampleLevel("))
    }
}
