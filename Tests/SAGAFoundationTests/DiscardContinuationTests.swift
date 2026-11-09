import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let discardContinuationShader = """
    @group(0) @binding(0) var<storage, read_write> counts: array<atomic<u32>, 4>;

    fn record_and_discard(lane: u32, x: f32) -> f32 {
        atomicAdd(&counts[lane], 1u);
        discard;
        atomicAdd(&counts[lane], 4u);
        return x * 4.0;
    }

    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point, 0.5, 1.0);
    }

    @fragment fn fs(@builtin(position) point: vec4f) -> @location(0) vec4f {
        let lane = u32(point.y) * 2u + u32(point.x);
        atomicAdd(&counts[lane], 2u);
        var derivativeInput = point.x * 2.0;
        if point.x < 1.0 { derivativeInput = record_and_discard(lane, point.x); }
        let delta = dpdxFine(derivativeInput);
        atomicAdd(&counts[lane], 7u + u32(delta));
        return vec4f(0.0, delta, 0.0, 1.0);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func discardAllowsFollowingStatementsInAReachableHelper(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(discardContinuationShader)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: format)
    guard case .text(_, let fragmentText) = fragment.artifact, case .text(_, let vertexText) = vertex.artifact else {
        Issue.record("Expected shader text")
        return
    }
    switch format {
    case .wgsl:
        #expect(fragmentText.contains("discard;"))
        _ = try compiler.parseAndValidate(.wgsl(fragmentText))
    case .msl:
        #expect(fragmentText.contains("discard_fragment();"))
    case .hlsl:
        #expect(fragmentText.contains("discard;"))
    case .glsl:
        #expect(fragmentText.contains("#extension GL_EXT_demote_to_helper_invocation : require"))
        #expect(fragmentText.contains("demote;"))
        #expect(!vertexText.contains("GL_EXT_demote_to_helper_invocation"))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [false, true])
func discardCanAppearInContinuingAndFallsThroughToTheNextStatement(_ roundTrip: Bool) throws {
    var source = ShaderInput.wgsl("""
        @fragment fn fs() -> @location(0) vec4f {
            var count = 0u;
            loop {
                count += 1u;
                continuing {
                    discard;
                    count += 1u;
                    break if count >= 2u;
                }
            }
            return vec4f(f32(count));
        }
        """)
    let compiler = IndependentShaderCompiler()
    if roundTrip {
        guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else {
            throw ShaderFailure.incompatibleArtifact
        }
        source = .spirv(words)
    }
    for target in [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl] {
        let stage = try compiler.compileRenderStage(source, entryPoint: "fs", to: target)
        if target == .wgsl, case .text(_, let text) = stage.artifact {
            _ = try compiler.parseAndValidate(.wgsl(text))
        }
    }
}

@Test(arguments: [
    "@fragment fn missing_return() -> @location(0) vec4f { discard; }",
    "fn discard_fragment() { discard; } @compute @workgroup_size(1) fn run() { discard_fragment(); }",
    "fn discard_fragment() { discard; } @vertex fn run() -> @builtin(position) vec4f { discard_fragment(); return vec4f(0.0); }",
])
func discardPreservesReturnAndStageValidation(source: String) {
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func discardCommitsOnlyEarlierWritesAndSuppressesRasterOutputOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(discardContinuationShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let counts = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: pipeline, target: color, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)],
                storageBuffers: [.init(binding: 0): .init(buffer: counts)]),
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
        .copy(source: counts, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]
    let result = try await device.submitCommands(commands).wait()
    let pixels = result[0]
    #expect([pixels[0], pixels[1], pixels[64], pixels[65]] == [0xffff0000, 0xff00ff00, 0xffff0000, 0xff00ff00])
    #expect(result[1] == [3, 10, 3, 10])
    await device.close()
}
#endif
