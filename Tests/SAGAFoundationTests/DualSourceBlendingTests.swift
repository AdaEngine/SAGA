import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let dualSourceShader = """
    enable dual_source_blending;
    struct Outputs {
        @location(0) @blend_src(1) weight: vec4f,
        @blend_src(0) @location(0) color: vec4f,
    }
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f { return vec4f(point, 0.5, 1.0); }
    @fragment fn fs() -> Outputs {
        return Outputs(vec4f(0.5, 0.25, 0.75, 0.25), vec4f(1.0, 0.5, 0.25, 1.0));
    }
    """

private func dualSourceInput(_ compiler: IndependentShaderCompiler, roundTrip: Bool) throws -> ShaderInput {
    let source = ShaderInput.wgsl(dualSourceShader)
    guard roundTrip else { return source }
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return .spirv(words)
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func dualSourceFragmentOutputsRemainSeparateFromColorAttachments(format: ShaderOutputFormat, roundTrip: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileRenderStage(dualSourceInput(compiler, roundTrip: roundTrip), entryPoint: "fs", to: format)
    #expect(artifact.usesDualSourceBlending)
    #expect(artifact.colorOutputKinds == [0: .float])
    guard case .text(_, let source) = artifact.artifact else { Issue.record("Expected shader text"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("enable dual_source_blending;"))
        #expect(source.contains("@blend_src(0)") && source.contains("@blend_src(1)"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.dualSourceBlending])
    case .msl: #expect(source.contains("color(0) index(0)") && source.contains("color(0) index(1)"))
    case .hlsl: #expect(source.contains("SV_Target0") && source.contains("SV_Target1"))
    case .glsl:
        #expect(source.contains("layout(location = 0, index = 0)") && source.contains("layout(location = 0, index = 1)"))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    "struct S { @location(0) @blend_src(0) a: vec4f, }",
    "struct S { @location(0) @blend_src(1) a: vec4f, @location(0) @blend_src(1) b: vec4f, }",
    "struct S { @location(0) @blend_src(0) a: vec4f, @location(1) @blend_src(1) b: vec4f, }",
    "struct S { @location(0) @blend_src(0) a: vec4f, @location(0) @blend_src(1) b: vec3f, }",
    "struct S { @location(0) @blend_src(0) a: vec4f, @location(0) @blend_src(1) b: vec4f, @location(1) c: vec4f, }",
    "struct S { @blend_src(0) a: vec4f, @location(0) @blend_src(1) b: vec4f, }",
    "@fragment fn fs() -> @location(0) @blend_src(0) vec4f { return vec4f(0.0); }",
    """
    struct S { @location(0) @blend_src(0) a: vec4f, @location(0) @blend_src(1) b: vec4f, }
    @fragment fn fs(input: S) -> @location(0) vec4f { return input.a; }
    """,
])
func rejectsInvalidDualSourceBindingsEvenOnUnusedStructures(source: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl("enable dual_source_blending;\n" + source),
                                                          capabilities: [.dualSourceBlending])
    }
}

@Test func dualSourceScalarOutputsTranslateAndRequireBothEnableAndCapability() throws {
    let compiler = IndependentShaderCompiler()
    let source = """
        enable dual_source_blending;
        struct Outputs { @location(0) @blend_src(0) color: f32, @location(0) @blend_src(1) weight: f32, }
        @fragment fn fs() -> Outputs { return Outputs(1.0, 0.5); }
        """
    for format in [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl] {
        _ = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: format)
    }
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(source)) }
    let noEnable = String(source.dropFirst("enable dual_source_blending;\n".count))
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl(noEnable), capabilities: [.dualSourceBlending])
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func blendsUsingBothFragmentSourcesOnMetal(_ roundTrip: Bool) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.dualSourceBlending])
    let compiler = IndependentShaderCompiler()
    let source = try dualSourceInput(compiler, roundTrip: roundTrip)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let ordinary = try compiler.compileRenderStage(.wgsl("""
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
        """), entryPoint: "fs", to: .msl)
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    let blend = RenderBlendState(
        color: .init(sourceFactor: .source1Color, destinationFactor: .oneMinusSource1Color),
        alpha: .init(sourceFactor: .source1Alpha, destinationFactor: .oneMinusSource1Alpha))
    let inverse = RenderBlendState(
        color: .init(sourceFactor: .oneMinusSource1Color, destinationFactor: .source1Color),
        alpha: .init(sourceFactor: .oneMinusSource1Alpha, destinationFactor: .source1Alpha))
    await #expect(throws: GPUError.self) {
        try await device.createRenderPipeline(vertex: vertex, fragment: ordinary, colorFormat: .rgba8Unorm,
            vertexBuffers: [layout], colorState: .init(blend: blend))
    }
    await #expect(throws: GPUError.self) {
        try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            vertexBuffers: [layout], colorState: .init(blend: blend), additionalColorTargets: [1: .init(format: .rgba8Unorm)])
    }
    let direct = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], colorState: .init(blend: blend))
    let reverse = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], colorState: .init(blend: inverse))
    let noBlend = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout])
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 256, usage: [.mapRead, .copyDestination]))
    let clear = RenderClearColor(red: 0, green: 1, blue: 0, alpha: 0)
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]
    var commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
    ]
    for pipeline in [direct, reverse, noBlend] {
        commands.append(.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                                vertexBuffers: [.init(buffer: vertices)]))
        commands += capture
    }
    let pixels = try await device.submitCommands(commands).wait().flatMap { $0 }
    let expected = [[128, 223, 48, 64], [128, 159, 16, 191], [255, 128, 64, 255]]
    #expect(pixels.count == expected.count)
    for (pixel, channels) in zip(pixels, expected) {
        for (lane, value) in channels.enumerated() {
            #expect(abs(Int((pixel >> (lane * 8)) & 255) - value) <= 1)
        }
    }
    await device.close()
}
#endif
