import Testing
import SAGACompiler
import SAGAShaderIR

@Test func publicMeshCompilerRoutesGLSLAndSelectsEntryPoint() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output {
        @builtin(vertices) vertices: array<Vertex, 3>,
        @builtin(primitives) primitives: array<Primitive, 1>,
        @builtin(vertex_count) vertex_count: u32,
        @builtin(primitive_count) primitive_count: u32,
    }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(1) fn mesh() {
        output.vertex_count = 3u;
        output.primitive_count = 1u;
        output.vertices[0].position = vec4f(0.0, 0.0, 0.0, 1.0);
        output.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl)
    guard case .text(let format, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(format == .glsl)
    #expect(code.contains("GL_EXT_mesh_shader"))
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMeshStage(.wgsl(source), entryPoint: "missing", to: .glsl)
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .wgsl)
    }
}
