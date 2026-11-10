import Testing
import SAGACompiler
import SAGAShaderIR

@Test func publicMeshCompilerGeneratesBothNativeTargets() throws {
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
    @mesh(output) @workgroup_size(4) fn main(@builtin(local_invocation_index) lane: u32) {
        if lane == 0u {
            output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(0.0, 0.0, 0.0, 1.0);
            output.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
    }
    """
    let reflected = try IndependentShaderCompiler().compileReflectedMeshStage(.wgsl(source), entryPoint: "main", to: .msl)
    #expect(reflected.sourceEntryPoint == "main")
    #expect(reflected.entryPoint == "saga_main")
    #expect(reflected.workgroupSize == IRWorkgroupSize(x: 4))
    #expect(reflected.topology == .triangles)
    #expect(reflected.maxVertices == 3)
    #expect(reflected.maxPrimitives == 1)
    #expect(reflected.payload == nil)
    #expect(reflected.vertexVaryings.isEmpty)
    for format in [ShaderOutputFormat.msl, .hlsl] {
        let artifact = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "main", to: format)
        guard case .text(let actual, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(actual == format)
        #expect(code.contains(format == .msl ? "set_primitive_count" : "SetMeshOutputCounts"))
    }
    let glsl = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "main", to: .glsl)
    guard case .text(let format, let code) = glsl else { throw ShaderFailure.incompatibleArtifact }
    #expect(format == .glsl)
    #expect(code.contains("GL_EXT_mesh_shader"))
}
