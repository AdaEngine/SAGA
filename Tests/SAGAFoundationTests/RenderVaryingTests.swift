import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

let varyingProgram = """
    struct VertexOutput {
        @location(2) tint: vec4f,
        @builtin(position) clip: vec4f,
        @location(5) @interpolate(perspective, centroid) gain: f32,
    }
    struct FragmentInput {
        @location(5) @interpolate(perspective, centroid) gain: f32,
        @location(2) tint: vec4f,
    }
    fn makeVertex(tint: vec4f, clip: vec4f) -> VertexOutput {
        var output: VertexOutput;
        output.tint = tint;
        output.clip = clip;
        output.gain = 1.0;
        return output;
    }
    @vertex fn vertexMain(@builtin(vertex_index) id: u32) -> VertexOutput {
        if id == 0u { return makeVertex(vec4f(1.0, 0.0, 0.0, 1.0), vec4f(-1.0, 1.0, 0.25, 1.0)); }
        if id == 1u { return makeVertex(vec4f(0.0, 1.0, 0.0, 1.0), vec4f(2.0, 2.0, 0.5, 2.0)); }
        return makeVertex(vec4f(0.0, 0.0, 1.0, 1.0), vec4f(-4.0, -4.0, 1.0, 4.0));
    }
    @fragment fn fragmentMain(input: FragmentInput) -> @location(0) vec4f {
        return shade(input.tint, input.gain);
    }
    fn shade(tint: vec4f, gain: f32) -> vec4f {
        var color = tint;
        let before = color;
        color.r = 0.0;
        return vec4f(before.r * gain, color.g, color.b, 1.0);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsVaryingsByLocationAcrossStageStructs(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(varyingProgram), entryPoint: "vertexMain", to: format)
    let fragment = try compiler.compileRenderStage(.wgsl(varyingProgram), entryPoint: "fragmentMain", to: format)
    #expect(vertex.varyings == fragment.varyings)
    #expect(vertex.varyings.map(\.location) == [2, 5])
    #expect(vertex.varyings.map(\.componentCount) == [4, 1])
    #expect(vertex.varyings.map(\.sampling) == [.center, .centroid])
    guard case .text(_, let vs) = vertex.artifact, case .text(_, let fs) = fragment.artifact else {
        Issue.record("Expected generated text")
        return
    }
    switch format {
    case .msl:
        #expect(vs.contains("user(locn2)"))
        #expect(fs.contains("user(locn5), centroid_perspective"))
    case .hlsl:
        #expect(vs.contains("TEXCOORD2"))
        #expect(fs.contains("centroid float") && fs.contains("TEXCOORD5"))
    case .glsl:
        #expect(vs.contains("layout(location = 2) out vec4 saga_varying_2"))
        #expect(fs.contains("layout(location = 5) centroid in float saga_varying_5"))
        let parsedVertex = try compiler.parseAndValidate(.glsl(vs, stage: .vertex, defines: [:])).module
        let vertexEntry = try #require(parsedVertex.entryPoints.first)
        let vertexResult = try #require(parsedVertex.functions.value(for: vertexEntry.function).result)
        guard case .structure(let members, _) = try parsedVertex.types.value(for: vertexResult.type).kind else {
            Issue.record("Expected GLSL vertex outputs to lower to a structure")
            return
        }
        let parsedFragment = try compiler.parseAndValidate(.glsl(fs, stage: .fragment, defines: [:])).module
        let fragmentEntry = try #require(parsedFragment.entryPoints.first)
        let fragmentBindings = try parsedFragment.functions.value(for: fragmentEntry.function).arguments.compactMap(\.binding)
        #expect(members.count == 3)
        #expect(members.contains { $0.binding == .builtIn(.position(invariant: false)) })
        #expect(fragmentBindings.count == 2)
        #expect(fragmentBindings.allSatisfy { binding in members.contains { $0.binding == binding } })
    case .wgsl:
        #expect(vs.contains("@location(2)"))
        #expect(fs.contains("@location(5) @interpolate(perspective, centroid)"))
        #expect(try compiler.parseAndValidate(.wgsl(vs)).module.entryPoints.first?.stage == .vertex)
        #expect(try compiler.parseAndValidate(.wgsl(fs)).module.entryPoints.first?.stage == .fragment)
    default: Issue.record("Unexpected format")
    }
}

@Test func rejectsInterpolationOnNonInterstageWGSLBindings() {
    let compiler = IndependentShaderCompiler()
    for source in [
        "@vertex fn f(@location(0) @interpolate(perspective) v: vec4f) -> @builtin(position) vec4f { return v; }",
        "@fragment fn f() -> @location(0) @interpolate(perspective, centroid) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }",
        "@fragment fn f(@interpolate(perspective) v: f32) -> @location(0) f32 { return v; }",
        "@fragment fn f(@location(0) @interpolate(perspective, either) v: f32) -> @location(0) f32 { return v; }",
    ] {
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(source)) }
    }
}

@Test func compilesDirectLocationInputsForFragmentStage() throws {
    let source = """
        @fragment fn run(@location(0) red: f32, @location(1) channels: vec2f) -> @location(0) vec4f {
            return vec4f(red, channels, 1.0);
        }
        """
    let compiler = IndependentShaderCompiler()
    for format: ShaderOutputFormat in [.msl, .hlsl, .glsl, .wgsl] {
        let shader = try compiler.compileRenderStage(.wgsl(source), entryPoint: "run", to: format)
        #expect(shader.varyings.map(\.componentCount) == [1, 2])
    }
}

#if canImport(Metal)
@Test func rasterizesPerspectiveVaryingsThroughSAGAQueueOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(.wgsl(varyingProgram), entryPoint: "vertexMain", to: .msl)
        let fragment = try compiler.compileRenderStage(.wgsl(varyingProgram), entryPoint: "fragmentMain", to: .msl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                           usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 1824, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: texture, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 1824),
        ])
        let result = try await ticket.wait()
        let pixels = try #require(result.first)
        try #require(pixels.count == 456)
        for y in 0..<8 {
            for x in 0..<8 where x + y < 7 {
                let u = (Double(x) + 0.5) / 8
                let v = (Double(y) + 0.5) / 8
                let weights = [1 - u - v, u / 2, v / 4]
                let sum = weights.reduce(0, +)
                let pixel = pixels[y * 64 + x]
                for (channel, weight) in weights.enumerated() {
                    let expected = Int((255 * weight / sum).rounded())
                    #expect(abs(Int((pixel >> (8 * channel)) & 255) - expected) <= 1)
                }
                #expect(pixel >> 24 == 255)
            }
        }
        #expect(pixels.last == 0xff000000)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
