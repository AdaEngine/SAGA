import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvDemoteInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read_write> counts: array<atomic<u32>, 4>;

    fn recordAndDiscard(lane: u32, x: f32) -> f32 {
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
        if point.x < 1.0 { derivativeInput = recordAndDiscard(lane, point.x); }
        let delta = dpdxFine(derivativeInput);
        atomicAdd(&counts[lane], 7u + u32(delta));
        return vec4f(0.0, delta, 0.0, 1.0);
    }
    """)

private func exportDemotion(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvDemoteInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvDemoteCodeGeneratorPreservesTheHelperContinuation() throws {
    let words = try exportDemotion(IndependentShaderCompiler())
    try #require(words.count > 5)
    #expect(words[1] == 0x00010300)
    var extensions: Set<String> = []
    var capabilities: Set<UInt32> = []
    var opcodes: Set<UInt32> = []
    var demotions = 0
    var demotedFunction = false
    var atomicAfterDemotion = false
    var returnAfterDemotion = false
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        switch opcode {
        case 10: // OpExtension
            let bytes = words[(offset + 1)..<(offset + count)].flatMap { word in
                (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) }
            }
            extensions.insert(String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self))
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 56: demotedFunction = false // OpFunctionEnd
        case 5380 where count == 1: // OpDemoteToHelperInvocationEXT
            demotedFunction = true
            demotions += 1
        case 234 where demotedFunction: atomicAfterDemotion = true
        case 254 where demotedFunction: returnAfterDemotion = true
        default: break
        }
        offset += count
    }
    #expect(extensions.contains("SPV_EXT_demote_to_helper_invocation"))
    #expect(capabilities.contains(5379))
    #expect(demotions == 1 && atomicAfterDemotion && returnAfterDemotion)
    #expect(!opcodes.contains(252)) // OpKill would terminate the helper.
    #expect(opcodes.contains(210)) // OpDPdxFine
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvDemoteCodeGeneratorRoundTripsLinkedRenderStages(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(try exportDemotion(compiler))
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: format)
    #expect(fragment.storageBindings.map(\.minimumByteSize) == [16])
    guard case .text(_, let vertexText) = vertex.artifact,
          case .text(_, let fragmentText) = fragment.artifact else { throw ShaderFailure.incompatibleArtifact }
    switch format {
    case .wgsl:
        #expect(fragmentText.contains("discard;"))
        _ = try compiler.parseAndValidate(.wgsl(vertexText))
        _ = try compiler.parseAndValidate(.wgsl(fragmentText))
    case .msl: #expect(fragmentText.contains("discard_fragment();"))
    case .hlsl: #expect(fragmentText.contains("discard;"))
    case .glsl:
        #expect(fragmentText.contains("GL_EXT_demote_to_helper_invocation") && fragmentText.contains("demote;"))
        #expect(!vertexText.contains("GL_EXT_demote_to_helper_invocation"))
    default: Issue.record("Unexpected text format")
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvDemoteCodeGeneratorPreservesPixelsAndStorageSuppressionOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let input = try roundTrip ? ShaderInput.spirv(exportDemotion(compiler)) : spirvDemoteInput
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let counts = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                        usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let snapshots = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            .render(pipeline: pipeline, target: color, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)],
                    storageBuffers: [.init(binding: 0): .init(buffer: counts)]),
            .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 512),
            .copy(source: counts, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        try #require(snapshots.count == 2 && snapshots[0].count == 128)
        #expect([0, 1, 64, 65].map { snapshots[0][$0] } == [0xffff0000, 0xff00ff00, 0xffff0000, 0xff00ff00])
        #expect(snapshots[1] == [3, 10, 3, 10])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
