import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvPackingInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 6>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 9>;

    @compute @workgroup_size(1) fn run() {
        let normalized = bitcast<vec4f>(input[0]);
        let halfValues = bitcast<vec2f>(input[1].xy);
        let word = input[1].z;
        let signed = bitcast<vec4i>(input[2]);
        let unsigned = input[3];
        let channels = input[4];
        let dotOperands = input[5];
        output[0] = vec4u(pack4x8snorm(normalized), pack4x8unorm(normalized),
                          pack2x16snorm(normalized.yz), pack2x16unorm(normalized.zw));
        output[1] = vec4u(pack4xI8(signed), pack4xI8Clamp(signed), pack4xU8(unsigned), pack4xU8Clamp(unsigned));
        output[2] = bitcast<vec4u>(unpack4xI8(word));
        output[3] = unpack4xU8(word);
        output[4] = bitcast<vec4u>(unpack4x8snorm(channels.x));
        output[5] = bitcast<vec4u>(unpack4x8unorm(channels.y));
        output[6] = vec4u(bitcast<vec2u>(unpack2x16snorm(channels.z)), bitcast<vec2u>(unpack2x16unorm(channels.w)));
        let decoded = unpack2x16float(input[1].w);
        output[7] = vec4u(pack2x16float(halfValues), bitcast<vec2u>(decoded), pack2x16float(decoded));
        output[8] = vec4u(bitcast<u32>(dot4I8Packed(word, dotOperands.x)), dot4U8Packed(word, dotOperands.x),
                          bitcast<u32>(dot4I8Packed(dotOperands.y, dotOperands.z)), dot4U8Packed(dotOperands.y, dotOperands.z));
    }
    """)

private func exportPacking(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvPackingInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvPackingCodeGeneratorUsesExactNormalizedRoundingAndPortableDots() throws {
    let words = try exportPacking(IndependentShaderCompiler())
    var capabilities: Set<UInt32> = []
    var extendedOpcodes: Set<UInt32> = []
    var integerAdds = 0
    var integerMultiplies = 0
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 12 where count >= 5: extendedOpcodes.insert(words[offset + 4])
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 128: integerAdds += 1
        case 132: integerMultiplies += 1
        default: break
        }
        offset += count
    }
    #expect(extendedOpcodes.isSuperset(of: [58, 60, 61, 62, 63, 64])) // Half packing and all float unpacks.
    #expect(extendedOpcodes.isDisjoint(with: 54...57)) // WGSL normalized packs define tie rounding.
    #expect(capabilities.isDisjoint(with: [39, 6016, 6017, 6018, 6019])) // Int8 and integer-dot capabilities.
    #expect(integerMultiplies >= 16 && integerAdds >= 12) // Four four-lane dot products.
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvPackingCodeGeneratorRoundTripsNativeShaderFormats(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportPacking(compiler)), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [96, 144])
    #expect(!shader.requiresShaderF16) // Packed half values are represented in f32 and u32.
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvPackingCodeGeneratorPreservesPackedWordsAndSignedDotOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shaderInput = try roundTrip ? ShaderInput.spirv(exportPacking(compiler)) : spirvPackingInput
    let shader = try compiler.compileCompute(shaderInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 96, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 144, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 144, usage: [.mapRead, .copyDestination]))
        let seeds: [UInt32] = [
            Float(-2).bitPattern, Float(-0.5).bitPattern, Float(0.5).bitPattern, Float(2).bitPattern,
            Float(1).bitPattern, Float(-2).bitPattern, 0x7f00ff80, 0xc0003c00,
            UInt32(bitPattern: -129), .max, 127, 128,
            256, 255, 511, 0x12345678,
            0x7f008081, 0xff00ff00, 0x7fff8000, 0xffff0000,
            0x01020304, 0x80808080, .max, 0,
        ]
        let expected: [UInt32] = [
            0x7f40c181, 0xff800000, 0x4000c001, 0xffff8000,
            0x807fff7f, 0x7f7fff80, 0x78ffff00, .max,
            0xffffff80, .max, 0, 127,
            128, 255, 0, 127,
            0xbf800000, 0xbf800000, 0, 0x3f800000,
            0, 0x3f800000, 0, 0x3f800000,
            0xbf800000, 0x3f800000, 0, 0x3f800000,
            0xc0003c00, 0x3f800000, 0xc0000000, 0xc0003c00,
            UInt32(bitPattern: -388), 1404, 512, 130560,
        ]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: seeds),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 144), .read(buffer: readback, byteCount: 144),
        ]).wait()
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
