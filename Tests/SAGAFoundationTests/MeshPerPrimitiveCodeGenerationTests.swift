import SAGACompiler
import SAGAShaderIR
import Testing

// Fragment interface adapted from naga/tests/in/wgsl/mesh-shader.wgsl at the pinned revision.
private let perPrimitiveFragment = """
    enable wgpu_mesh_shader;
    struct Input { @per_primitive @location(1) color: vec4f, }
    @fragment fn fs(input: Input) -> @location(0) vec4f { return input.color; }
    """

@Test func perPrimitiveFragmentCodeGeneration() throws {
    let compiler = IndependentShaderCompiler()
    func source(_ format: ShaderOutputFormat) throws -> String {
        let result = try compiler.compileRenderStage(.wgsl(perPrimitiveFragment), entryPoint: "fs", to: format)
        guard case .text(_, let source) = result.artifact else { throw ShaderFailure.unsupported("expected text") }
        return source
    }
    #expect(try source(.msl).contains("[[user(locn1), flat]]"))
    #expect(try source(.hlsl).contains(": TEXCOORD1 : primitive"))
    let wgsl = try source(.wgsl)
    #expect(wgsl.contains("@per_primitive"))
    _ = try compiler.parseAndValidate(.wgsl(wgsl), capabilities: [.meshShader])
    let glsl = try source(.glsl)
    #expect(glsl.contains("#extension GL_EXT_mesh_shader : require"))
    #expect(glsl.contains("layout(location = 1) perprimitiveEXT in vec4 saga_varying_1;"))
    #expect(!glsl.contains("flat perprimitiveEXT"))
}
