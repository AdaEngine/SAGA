import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvInt64BitInput = ShaderInput.wgsl("""
    struct Operands { bits: vec4<u64>, offsets: vec4u, bounds: vec4u }
    @group(0) @binding(0) var<storage, read> input: Operands;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4<u64>, 16>;

    @compute @workgroup_size(1) fn run() {
        let bits = input.bits;
        let signed = bitcast<vec4<i64>>(bits);
        let offsets = input.offsets;
        let bounds = input.bounds;
        output[0] = countLeadingZeros(bits);
        output[1] = bitcast<vec4<u64>>(countTrailingZeros(signed));
        output[2] = bitcast<vec4<u64>>(countOneBits(signed));
        output[3] = reverseBits(bits);
        output[4] = firstLeadingBit(bits);
        output[5] = bitcast<vec4<u64>>(firstLeadingBit(signed));
        output[6] = bitcast<vec4<u64>>(firstTrailingBit(signed));
        output[7] = bitcast<vec4<u64>>(extractBits(signed, offsets.y, bounds.w));
        output[8] = insertBits(bits, vec4<u64>(0xfedcba987654lu), offsets.y, bounds.w);
        output[9] = extractBits(bits, offsets.z, bounds.y);
        output[10] = bitcast<vec4<u64>>(insertBits(signed, vec4<i64>(-1li), offsets.w, bounds.y));
        output[11] = bitcast<vec4<u64>>(extractBits(signed, offsets.x, bounds.y));
        output[12] = insertBits(bits, vec4<u64>(0xfffffffffffffffflu), bounds.x, bounds.y);
        output[13] = vec4<u64>(extractBits(bits.w, bounds.z, bounds.y),
                               bitcast<u64>(extractBits(signed.z, offsets.w, bounds.y)),
                               bitcast<u64>(insertBits(signed.x, signed.y, offsets.x, bounds.y)),
                               insertBits(bits.y, bits.w, bounds.z, bounds.y));
        output[14] = vec4<u64>(u64(countLeadingZeros(signed.x)), countTrailingZeros(bits.x),
                               u64(countOneBits(signed.w)), bitcast<u64>(reverseBits(signed.y)));
        output[15] = vec4<u64>(bitcast<u64>(firstLeadingBit(signed.z)), firstLeadingBit(bits.z),
                               firstTrailingBit(bits.x), bitcast<u64>(firstTrailingBit(signed.z)));
    }
    """)

private func exportInt64BitBuiltins(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvInt64BitInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvInt64BitBuiltinsUseVulkanWordSizedInstructions() throws {
    let words = try exportInt64BitBuiltins(IndependentShaderCompiler())
    var integerWidths: [UInt32: UInt32] = [:]
    var wordOperationTypes: [UInt32] = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 21 where count == 4: integerWidths[words[offset + 1]] = words[offset + 2] // OpTypeInt
        case 23 where count == 4: integerWidths[words[offset + 1]] = integerWidths[words[offset + 2]] // OpTypeVector
        case 12 where count >= 5 && (73...75).contains(words[offset + 4]): // GLSL.std.450 Find*Bit
            wordOperationTypes.append(words[offset + 1])
        case 201...205 where count >= 4: wordOperationTypes.append(words[offset + 1])
        default: break
        }
        offset += count
    }
    #expect(integerWidths.values.contains(64))
    try #require(!wordOperationTypes.isEmpty)
    #expect(wordOperationTypes.allSatisfy { integerWidths[$0] == 32 })
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvInt64BitBuiltinsRoundTripNativeShaderFormats(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportInt64BitBuiltins(compiler)), to: format)
    #expect(shader.requiresShaderInt64)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64, 512])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.shaderInt64))
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvInt64BitBuiltinsMatchCPUAcrossWordBoundariesOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shaderInput = try roundTrip ? ShaderInput.spirv(exportInt64BitBuiltins(compiler)) : spirvInt64BitInput
    let shader = try compiler.compileCompute(shaderInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt64])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 512, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 512, usage: [.mapRead, .copyDestination]))
        func words(_ values: [UInt64]) -> [UInt32] {
            values.flatMap { [UInt32(truncatingIfNeeded: $0), UInt32($0 >> 32)] }
        }
        func leadingIndex(_ value: UInt64) -> UInt64 {
            value == 0 ? .max : UInt64(63 - value.leadingZeroBitCount)
        }
        func trailingIndex(_ value: UInt64) -> UInt64 {
            value == 0 ? .max : UInt64(value.trailingZeroBitCount)
        }
        func reversed(_ value: UInt64) -> UInt64 {
            (0..<64).reduce(0) { ($0 << 1) | ((value >> $1) & 1) }
        }
        let bits: [UInt64] = [0, 0x0000_0481_8000_0001, 1 << 63, .max]
        let crossWordMask: UInt64 = 0xfff
        let crossWordSign: UInt64 = 0x800
        let expected: [[UInt64]] = [
            bits.map { UInt64($0.leadingZeroBitCount) }, bits.map { UInt64($0.trailingZeroBitCount) },
            bits.map { UInt64($0.nonzeroBitCount) }, bits.map(reversed), bits.map(leadingIndex),
            bits.map { leadingIndex(Int64(bitPattern: $0) < 0 ? ~$0 : $0) }, bits.map(trailingIndex),
            bits.map { ((($0 >> 31) & crossWordMask) ^ crossWordSign) &- crossWordSign },
            bits.map { ($0 & ~(crossWordMask << 31)) | ((0xfedcba987654 & crossWordMask) << 31) },
            bits.map { $0 >> 32 }, bits.map { $0 | (1 << 63) }, bits, bits,
            [0, .max, bits[1], bits[1]], [64, 64, 64, reversed(bits[1])], [62, 63, .max, 63],
        ]
        let seeds = words(bits) + [0, 31, 32, 63, 64, .max, .max, 12]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: seeds),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 512), .read(buffer: readback, byteCount: 512),
        ]).wait()
        #expect(snapshots == [words(expected.flatMap { $0 })])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
