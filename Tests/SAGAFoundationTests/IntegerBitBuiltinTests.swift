import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let integerBitShader = """
    const edges = vec4u(countLeadingZeros(0u), countTrailingZeros(0u), countOneBits(0xffffffffu), reverseBits(1u));
    const_assert firstLeadingBit(-1i) == -1i;
    const_assert firstLeadingBit(-2147483647i - 1i) == 30i;
    const_assert firstTrailingBit(0u) == 0xffffffffu;
    const_assert extractBits(vec2i(-1, 8), 0u, 4u).y == -8i;
    const_assert insertBits(7u, 9u, 32u, 0u) == 7u;
    const_assert insertBits(vec2(0, 1), vec2u(3, 2), 28u, 4u).y == 0x20000001u;
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 2>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 14>;
    @compute @workgroup_size(1) fn run() {
        let bits = input[0];
        let signed = bitcast<vec4i>(bits);
        let ranges = input[1];
        output[0] = countLeadingZeros(bits);
        output[1] = bitcast<vec4u>(countTrailingZeros(signed));
        output[2] = countOneBits(bits);
        output[3] = bitcast<vec4u>(reverseBits(signed));
        output[4] = firstLeadingBit(bits);
        output[5] = bitcast<vec4u>(firstLeadingBit(signed));
        output[6] = bitcast<vec4u>(firstTrailingBit(signed));
        output[7] = extractBits(bits, ranges.x, ranges.y);
        output[8] = bitcast<vec4u>(extractBits(signed, ranges.x, ranges.y));
        output[9] = insertBits(bits, vec4u(5u), ranges.x, ranges.y);
        output[10] = bitcast<vec4u>(extractBits(signed, 0u, ranges.w));
        output[11] = bitcast<vec4u>(insertBits(signed, vec4i(-1), ranges.z, ranges.y));
        output[12] = vec4u(extractBits(bits.w, ranges.z, ranges.y), firstTrailingBit(bits.x),
                           bitcast<u32>(countLeadingZeros(signed.z)), insertBits(bits.x, 7, 0u, ranges.w));
        output[13] = edges;
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsIntegerBitBuiltinsWithTypedResults(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(integerBitShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("extractBits(") && source.contains("insertBits("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains("clz(") && source.contains("ctz(") && source.contains("reverse_bits("))
    case .hlsl:
        #expect(source.contains("firstbithigh(") && source.contains("firstbitlow(") && source.contains("countbits("))
    case .glsl:
        #expect(source.contains("findMSB(") && source.contains("findLSB(") && source.contains("bitfieldReverse("))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    ("const x = extractBits(1u, 32u, 1u);", ShaderDiagnosticCode.invalidType),
    ("const x = insertBits(1i, 1i, 0xffffffffu, 0xffffffffu);", .invalidType),
    ("const x = countOneBits(1.0);", .typeMismatch),
    ("const x = extractBits(1i, -1, 1u);", .invalidType),
    ("@compute @workgroup_size(1) fn run() { var x = 7u; let y = extractBits(x, 31u, 2u); }", .invalidType),
    ("@compute @workgroup_size(1) fn run() { var x = vec2u(7); let y = insertBits(x, vec3u(1), 0u, 1u); }", .typeMismatch),
    ("@compute @workgroup_size(1) fn run() { var x = 7u; let y = extractBits(x, 1i, 1u); }", .typeMismatch),
])
func rejectsInvalidIntegerBitBuiltins(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid integer bit builtin accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

@Test func boundsNestedConstantIntegerBitCalls() throws {
    func nested(_ count: Int) -> String {
        String(repeating: "countOneBits(", count: count) + "0u" + String(repeating: ")", count: count)
    }
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(.wgsl("const_assert \(nested(6)) == 0u;"))
    do {
        _ = try compiler.parseAndValidate(.wgsl("const x = \(nested(16));"))
        Issue.record("Deep constant bit calls did not reach the parser limit")
    } catch ShaderFailure.budgetExceeded(.depth) {}
}

private let expectedIntegerBitWords: [UInt32] = [
    32, 31, 0, 0, 32, 0, 31, 0, 0, 1, 1, 32, 0, 0x80000000, 1, 0xffffffff,
    0xffffffff, 0, 31, 31, 0xffffffff, 0, 30, 0xffffffff, 0xffffffff, 0, 31, 0,
    0, 0, 8, 15, 0, 0, 0xfffffff8, 0xffffffff, 0x50000000, 0x50000001, 0x50000000, 0x5fffffff,
    0, 1, 0x80000000, 0xffffffff, 0, 1, 0x80000000, 0xffffffff, 0, 0xffffffff, 0, 7,
    32, 32, 32, 0x80000000,
]

#if canImport(Metal)
@Test func executesIntegerBitBuiltinsAtWordBoundariesOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(integerBitShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 224, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 224, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0, 1, 0x80000000, 0xffffffff, 28, 8, 0xffffffff, 32]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 224), .read(buffer: read, byteCount: 224),
        ])
        #expect(try await ticket.wait() == [expectedIntegerBitWords])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
