import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvDerivativeInput = ShaderInput.wgsl("""
    struct Outputs {
        @location(0) regular: vec4u,
        @location(1) fine: vec4u,
        @location(2) coarse: vec4u,
    }

    @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        if index == 0u { return vec4f(-1, -1, 0, 1); }
        if index == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }

    fn encode(dx: f32, dy: f32, width: vec2f, flags: u32) -> vec4u {
        return vec4u(u32(dx), u32(8.0 + dy), u32(width.x), u32(width.y) + flags);
    }

    @fragment fn fs(@builtin(position) p: vec4f) -> Outputs {
        let q = vec2f(2.0 * p.x - 3.0 * p.y, 5.0 * p.y - 4.0 * p.x);
        let x = p.x < 1.0;
        let y = p.y < 1.0;
        let pair = vec2<bool>(x, y);
        let vectorFlags = select(0u, 16u, all(pair)) + select(0u, 32u, any(pair));
        let scalarFlags = select(0u, 16u, all(x)) + select(0u, 32u, any(y));
        let regular = encode(dpdx(q.x), dpdy(q.x), fwidth(q), vectorFlags);
        let fine = encode(dpdxFine(q).x, dpdyFine(q).x, fwidthFine(q), scalarFlags);
        let coarse = encode(dpdxCoarse(q).x, dpdyCoarse(q).x, fwidthCoarse(q), 0u);
        return Outputs(regular, fine, coarse);
    }
    """)

private func exportDerivativeModule(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvDerivativeInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func derivativeInstructions(_ words: [UInt32]) throws -> (capabilities: Set<UInt32>, opcodes: [UInt32: Int]) {
    var capabilities: Set<UInt32> = []
    var opcodes: [UInt32: Int] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes[opcode, default: 0] += 1
        if opcode == 17 && count == 2 { capabilities.insert(words[offset + 1]) }
        offset += count
    }
    return (capabilities, opcodes)
}

@Test func spirvDerivativeCodeGeneratorPreservesVariantsAndReducesOnlyVectors() throws {
    let words = try exportDerivativeModule(IndependentShaderCompiler())
    let (capabilities, opcodes) = try derivativeInstructions(words)
    #expect(capabilities.contains(51)) // DerivativeControl
    #expect((UInt32(207)...215).allSatisfy { opcodes[$0] == 1 })
    #expect(opcodes[154] == 1) // OpAny
    #expect(opcodes[155] == 1) // OpAll
}

@Test func spirvDerivativeCodeGeneratorOmitsDerivativeControlForDefaultVariants() throws {
    let input = ShaderInput.wgsl("""
        @fragment fn fs(@location(0) p: vec2f) -> @location(0) vec4f {
            return vec4f(dpdx(p.x), dpdy(p.y), fwidth(p));
        }
        """)
    guard case .spirv(let words) = try IndependentShaderCompiler().compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let (capabilities, opcodes) = try derivativeInstructions(words)
    #expect(!capabilities.contains(51))
    #expect((UInt32(207)...209).allSatisfy { opcodes[$0] == 1 })
    #expect((UInt32(210)...215).allSatisfy { opcodes[$0] == nil })
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvDerivativeCodeGeneratorRoundTripsLinkedRenderStages(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(try exportDerivativeModule(compiler))
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: format)
    #expect(fragment.colorOutputComponentCounts == [0: 4, 1: 4, 2: 4])
    if format == .wgsl {
        for shader in [vertex, fragment] {
            guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
            _ = try compiler.parseAndValidate(.wgsl(text))
        }
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvDerivativeCodeGeneratorPreservesExactPixelsOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let input = try roundTrip ? ShaderInput.spirv(exportDerivativeModule(compiler)) : spirvDerivativeInput
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
            additionalColorTargets: [1: .init(format: .rgba8Uint), 2: .init(format: .rgba8Uint)])
        let descriptor = TextureDescriptor(width: 2, height: 2, format: .rgba8Uint, usage: [.renderAttachment, .copySource])
        let regular = try await device.createTexture(descriptor)
        let fine = try await device.createTexture(descriptor)
        let coarse = try await device.createTexture(descriptor)
        let readback = try await device.createBuffer(.init(size: 264, usage: [.mapRead, .copyDestination]))
        var commands: [BufferCommand] = [
            .render(pipeline: pipeline, target: regular, clear: .transparentBlack, vertexCount: 3,
                    additionalColors: [1: .init(texture: fine, loadAction: .clear(.transparentBlack)),
                                       2: .init(texture: coarse, loadAction: .clear(.transparentBlack))]),
        ]
        for texture in [regular, fine, coarse] {
            commands += [.copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
                         .read(buffer: readback, byteCount: 264)]
        }
        let snapshots = try await device.submitCommands(commands).wait()
        try #require(snapshots.count == 3 && snapshots.allSatisfy { $0.count == 66 })
        let pixels = snapshots.map { snapshot in [0, 1, 64, 65].map { snapshot[$0] } }
        #expect(pixels == [[0x39050502, 0x29050502, 0x29050502, 0x09050502],
                           [0x39050502, 0x29050502, 0x19050502, 0x09050502],
                           Array(repeating: 0x09050502, count: 4)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
