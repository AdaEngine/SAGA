import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let int64BitShader = """
    const_assert countLeadingZeros(0lu) == 64lu;
    const_assert countTrailingZeros(0li) == 64li;
    const_assert countOneBits(0xfffffffffffffffflu) == 64lu;
    const_assert reverseBits(1lu) == 0x8000000000000000lu;
    const_assert firstLeadingBit(-9223372036854775808li) == 62li;
    const_assert firstTrailingBit(0lu) == 0xfffffffffffffffflu;
    const_assert extractBits(0x8000000000li, 28u, 12u) == -2048li;
    const_assert insertBits(vec2<u64>(1lu, 7lu), vec2<u64>(0xablu, 9lu), 28u, 8u).x == 0xab0000001lu;
    const_assert insertBits(7lu, 9lu, 64u, 0u) == 7lu;
    const_assert extractBits(0xfffffffffffffffflu, 0u, 64u) == 0xfffffffffffffffflu;

    struct Inputs { bits: vec4<u64>, ranges: vec4u, }
    @group(0) @binding(0) var<storage> input: Inputs;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4<u64>, 14>;
    @compute @workgroup_size(1) fn main() {
        let bits = input.bits;
        let signed = bitcast<vec4<i64>>(bits);
        let ranges = input.ranges;
        output[0] = countLeadingZeros(bits);
        output[1] = bitcast<vec4<u64>>(countTrailingZeros(signed));
        output[2] = countOneBits(bits);
        output[3] = bitcast<vec4<u64>>(reverseBits(signed));
        output[4] = firstLeadingBit(bits);
        output[5] = bitcast<vec4<u64>>(firstLeadingBit(signed));
        output[6] = bitcast<vec4<u64>>(firstTrailingBit(signed));
        output[7] = extractBits(bits, ranges.x, ranges.y);
        output[8] = bitcast<vec4<u64>>(extractBits(signed, ranges.x, ranges.y));
        output[9] = insertBits(bits, vec4<u64>(0xabclu), ranges.x, ranges.y);
        output[10] = bitcast<vec4<u64>>(extractBits(signed, 0u, ranges.w));
        output[11] = insertBits(bits, vec4<u64>(5lu), ranges.z, ranges.y);
        output[12] = vec4<u64>(firstTrailingBit(bits.x), countOneBits(bits.y),
                                reverseBits(bits.z), extractBits(bits.y, ranges.x, ranges.y));
        output[13] = bitcast<vec4<u64>>(insertBits(signed, vec4<i64>(-1li), 0u, ranges.w));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func int64BitBuiltinsCompileAndPreserveLayout(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(int64BitShader), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64, 448])
    guard case .text(_, let text) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.shaderInt64))
    } else {
        #expect(text.contains("64u"))
        #expect(text.contains(">> 32u"))
    }
}

@Test(arguments: [
    "const x = extractBits(1lu, 63u, 2u);",
    "const x = insertBits(1li, 1li, 65u, 0u);",
    "const x = insertBits(1li, 1lu, 0u, 1u);",
    "const x = extractBits(1lu, 0lu, 1u);",
    "@compute @workgroup_size(1) fn main() { var x = 1li; _ = extractBits(x, 63u, 2u); }",
])
func int64BitBuiltinsRejectInvalidTypesAndRanges(_ source: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.shaderInt64))
    }
}

@Test func int64BitBuiltinsCheckPipelineRangeAt64Bits() throws {
    let source = ShaderInput.wgsl("""
        override width: u32 = 32u;
        @group(0) @binding(0) var<storage> input: u64;
        @group(0) @binding(1) var<storage, read_write> output: array<u64, 2>;
        @compute @workgroup_size(1) fn main() {
            output[0] = extractBits(input, 32u, width);
            output[1] = insertBits(input, 0xfffffffffffffffflu, 32u, width);
        }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileCompute(source, to: .msl)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, to: .msl, constants: ["width": 33])
    }
}

#if canImport(Metal)
@Test func int64BitBuiltinsMatchCPUAcrossWordBoundariesOnMetal() async throws {
    let bits: [UInt64] = [0, 1 << 39, 1 << 63, .max]
    func leadingIndex(_ value: UInt64) -> UInt64 {
        value == 0 ? .max : UInt64(63 - value.leadingZeroBitCount)
    }
    func trailingIndex(_ value: UInt64) -> UInt64 {
        value == 0 ? .max : UInt64(value.trailingZeroBitCount)
    }
    func reversed(_ value: UInt64) -> UInt64 {
        (0..<64).reduce(0) { ($0 << 1) | ((value >> $1) & 1) }
    }
    func words(_ values: [UInt64]) -> [UInt32] {
        values.flatMap { [UInt32(truncatingIfNeeded: $0), UInt32($0 >> 32)] }
    }
    let fields = bits.map { ($0 >> 28) & 0xfff }
    let expected: [[UInt64]] = [
        bits.map { UInt64($0.leadingZeroBitCount) }, bits.map { UInt64($0.trailingZeroBitCount) },
        bits.map { UInt64($0.nonzeroBitCount) }, bits.map(reversed), bits.map(leadingIndex),
        bits.map { leadingIndex(Int64(bitPattern: $0) < 0 ? ~$0 : $0) }, bits.map(trailingIndex),
        fields, fields.map { ($0 ^ 0x800) &- 0x800 },
        bits.map { ($0 & ~(0xfff << 28)) | (0xabc << 28) }, bits, bits,
        [.max, 1, 1, 0x800], [.max, .max, .max, .max],
    ]
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(int64BitShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 448, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 448, usage: [.mapRead, .copyDestination]))
    let inputWords = words(bits) + [28, 12, .max, 64, 0, 0, 0, 0]
    let actual = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: inputWords),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 448), .read(buffer: readback, byteCount: 448),
    ]).wait()
    #expect(actual == [words(expected.flatMap { $0 })])
    await device.close()
}
#endif
