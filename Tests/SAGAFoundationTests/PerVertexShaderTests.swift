import SAGACompiler
import SAGAShaderIR
import Testing

private let perVertexScalar = ShaderInput.wgsl("""
    enable wgpu_per_vertex;

    @fragment
    fn fs_main(@location(0) @interpolate(per_vertex) v: array<f32, 3>) -> @location(0) vec4<f32> {
        return vec4(v[0], v[1], v[2], 1.0);
    }
    """)

private let perVertexStructured = ShaderInput.wgsl("""
    enable wgpu_per_vertex;

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) values: vec2f,
        @location(1) tint: f32,
    }
    struct FragmentInput {
        @location(0) @interpolate(per_vertex) values: array<vec2f, 3>,
        @location(1) tint: f32,
    }
    @vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOutput {
        let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        let i = f32(index);
        return VertexOutput(vec4f(points[index], 0, 1), vec2f(i, i + 1.0), 0.5);
    }
    fn color(input: FragmentInput) -> vec4f {
        return vec4f(input.values[0].x, input.values[1].y, input.values[2].x, input.tint);
    }
    @fragment fn fs(input: FragmentInput) -> @location(0) vec4f { return color(input); }
    """)

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .wgsl])
func perVertexShaderEmitsPinnedScalarInput(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(perVertexScalar, entryPoint: "fs_main", to: format)
    let varying = try #require(shader.varyings.first)
    #expect(varying.componentCount == 1 && varying.scalar == .f32)
    #expect(varying.interpolation == .perVertex && varying.sampling == .center)
    guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    switch format {
    case .msl:
        #expect(text.contains("metal::vertex_value<float>"))
        for index in ["first", "second", "third"] { #expect(text.contains(".get(metal::vertex_index::\(index))")) }
    case .hlsl:
        #expect(shader.hlslProfile == "ps_6_1" && text.contains("nointerpolation float"))
        let reads = text.split(separator: "\n").filter { $0.contains("GetAttributeAtVertex(") }
        #expect(reads.count == 3)
    case .wgsl:
        #expect(text.contains("enable wgpu_per_vertex;") && text.contains("@interpolate(per_vertex)"))
        #expect(text.contains("array<f32, 3>"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.perVertex])
    default: Issue.record("Unexpected output target")
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .wgsl], [false, true])
func perVertexShaderEmitsStructuredVectorInputAndMatchesVertexElement(_ format: ShaderOutputFormat, _ roundTrip: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input: ShaderInput
    if roundTrip {
        guard case .spirv(let words) = try compiler.compile(perVertexStructured, to: .spirv) else {
            throw ShaderFailure.incompatibleArtifact
        }
        input = .spirv(words)
    } else {
        input = perVertexStructured
    }
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: format)
    #expect(vertex.varyings.map(\.componentCount) == [2, 1])
    #expect(fragment.varyings.map(\.componentCount) == [2, 1])
    #expect(fragment.varyings.map(\.interpolation) == [.perVertex, .perspective])
    #expect(vertex.varyings.map(\.scalar) == fragment.varyings.map(\.scalar))
    guard case .text(_, let text) = fragment.artifact else { throw ShaderFailure.incompatibleArtifact }
    if format == .msl { #expect(text.contains("metal::vertex_value<float2>")) }
    if format == .hlsl { #expect(text.contains("nointerpolation float2") && text.contains("GetAttributeAtVertex(")) }
    if format == .wgsl {
        #expect(text.contains("array<vec2<f32>, 3>"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.perVertex])
    }
}

@Test func perVertexShaderRejectsGLSLAndWebGPUOutputs() throws {
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(perVertexScalar, capabilities: [.perVertex])
    #expect(throws: ShaderFailure.self) { try compiler.compileRenderStage(perVertexScalar, entryPoint: "fs_main", to: .glsl) }
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPURenderStage(perVertexScalar, entryPoint: "fs_main") }
}
