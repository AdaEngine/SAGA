import SAGACompiler
import SAGAShaderIR
import Testing

@Test func taskAndMeshGLSLDeclareInt64ImageExtension() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    @group(0) @binding(0) var image: texture_storage_2d<r64uint, atomic>;
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
      textureAtomicMax(image, vec2i(0), 7lu); payload.value = 1u; return vec3u(1u);
    }
    @mesh(out) @workgroup_size(1) fn mesh() {
      textureAtomicMax(image, vec2i(0), 7lu); out.vertex_count = 0u; out.primitive_count = 0u;
    }
    """
    let compiler = IndependentShaderCompiler()
    let artifacts = [try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl),
                     try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl)]
    for artifact in artifacts {
        guard case .text(_, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(code.contains("#extension GL_EXT_shader_image_int64 : require"))
        #expect(code.contains("u64image2D"))
    }
}
