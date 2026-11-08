import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private func clipShader(count: Int, varyingLocation: Int = 0) -> String {
    """
    enable clip_distances;
    struct Output {
        @builtin(clip_distances) planes: array<f32, \(count)>,
        @location(\(varyingLocation)) tint: f32,
        @builtin(position) position: vec4f,
    }

    fn projected(p: vec2f) -> Output {
        var result: Output;
        result.position = vec4f(p, 0.0, 1.0);
        result.tint = 1.0;
        result.planes[0] = p.x;
        \(count > 1 ? "result.planes[\(count - 1)] = p.y;" : "")
        return result;
    }

    @vertex fn vs(@location(0) point: vec2f) -> Output {
        return projected(point);
    }

    @fragment fn fs(@location(\(varyingLocation)) tint: f32) -> @location(0) vec4f {
        return vec4f(0.0, tint, 0.0, 1.0);
    }
    """
}

@Test(arguments: [1, 4, 5, 8], [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsClipDistancesForVertexOutputs(count: Int, format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(clipShader(count: count))
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
    #expect(vertex.clipDistanceCount == count)
    #expect(fragment.clipDistanceCount == 0)
    #expect(vertex.varyings.map(\.location) == [0])
    #expect(vertex.varyings == fragment.varyings)
    guard case .text(_, let vertexText) = vertex.artifact, case .text(_, let fragmentText) = fragment.artifact else {
        Issue.record("Expected text artifacts")
        return
    }
    switch format {
    case .wgsl:
        #expect(vertexText.contains("enable clip_distances;"))
        #expect(!fragmentText.contains("enable clip_distances;"))
        _ = try compiler.parseAndValidate(.wgsl(vertexText), capabilities: [.clipDistances])
        _ = try compiler.parseAndValidate(.wgsl(fragmentText))
    case .msl:
        #expect(vertexText.contains("[[clip_distance]] [\(count)]"))
    case .hlsl:
        #expect(vertexText.contains("SV_ClipDistance0"))
        #expect(vertexText.contains("SV_ClipDistance1") == (count > 4))
    case .glsl:
        #expect(vertexText.contains("out float gl_ClipDistance[\(count)];"))
        #expect(vertexText.contains("gl_ClipDistance[\(count - 1)] ="))
    default:
        Issue.record("Unexpected target")
    }
}

@Test func clipDistancesRequireTheExtensionAndCapability() throws {
    let compiler = IndependentShaderCompiler()
    let source = clipShader(count: 1)
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(source)) }
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl(String(source.dropFirst("enable clip_distances;\n".count))),
                                      capabilities: [.clipDistances])
    }
}

@Test(arguments: [
    """
    struct Output { @builtin(position) position: vec4f, @builtin(clip_distances) planes: array<f32, 9>, }
    @vertex fn vs() -> Output { return Output(); }
    """,
    """
    struct Output { @builtin(position) position: vec4f, @builtin(clip_distances) planes: array<i32, 2>, }
    @vertex fn vs() -> Output { return Output(); }
    """,
    """
    @vertex fn vs(@builtin(clip_distances) planes: array<f32, 2>) -> @builtin(position) vec4f {
        return vec4f(planes[0]);
    }
    """,
    """
    @fragment fn fs(@builtin(clip_distances) planes: array<f32, 2>) -> @location(0) vec4f {
        return vec4f(planes[0]);
    }
    """,
    """
    struct Output {
        @builtin(position) position: vec4f,
        @builtin(clip_distances) planes: array<f32, 1>,
        @builtin(clip_distances) duplicate: array<f32, 1>,
    }
    @vertex fn vs() -> Output { return Output(); }
    """,
])
func rejectsInvalidClipDistanceInterfaces(source: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl("enable clip_distances;\n" + source),
                                                         capabilities: [.clipDistances])
    }
}

#if canImport(Metal)
@Test func clipDistancesReserveVertexOutputLocations() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    let fits = ShaderInput.wgsl(clipShader(count: 4, varyingLocation: 14))
    let exceeds = ShaderInput.wgsl(clipShader(count: 5, varyingLocation: 14))
    let fragment = try compiler.compileRenderStage(fits, entryPoint: "fs", to: .msl)
    let validVertex = try compiler.compileRenderStage(fits, entryPoint: "vs", to: .msl)
    let invalidVertex = try compiler.compileRenderStage(exceeds, entryPoint: "vs", to: .msl)
    _ = try await device.createRenderPipeline(vertex: validVertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                               vertexBuffers: [layout])
    await #expect(throws: GPUError.validation("Vertex outputs exceed the inter-stage limit with clip distances")) {
        try await device.createRenderPipeline(vertex: invalidVertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                               vertexBuffers: [layout])
    }
    await device.close()
}

@Test(arguments: [1, 5, 8])
func clipsPrimitivesAgainstWGSLPlanesOnMetal(count: Int) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.clipDistances])
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(clipShader(count: count))
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)]),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
    ])
    let pixels = try #require(try await ticket.wait().first)
    let blue: UInt32 = 0xffff0000
    let green: UInt32 = 0xff00ff00
    #expect([pixels[0], pixels[1], pixels[64], pixels[65]] == [blue, green, blue, count == 1 ? green : blue])
    await device.close()
}
#endif
