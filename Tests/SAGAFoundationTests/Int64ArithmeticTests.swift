import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let int64ArithmeticShader = """
    struct Operands {
        signed_values: vec2<i64>,
        signed_divisors: vec2<i64>,
        unsigned_values: vec2<u64>,
        unsigned_divisors: vec2<u64>,
        counts: vec2u,
    }
    @group(0) @binding(0) var<storage> input: Operands;
    @group(0) @binding(1) var<storage, read_write> output: array<vec2<u64>, 12>;
    @compute @workgroup_size(1) fn main() {
        let a = input.signed_values;
        let b = input.signed_divisors;
        let x = input.unsigned_values;
        let y = input.unsigned_divisors;
        let counts = input.counts;
        output[0] = bitcast<vec2<u64>>(a / b);
        output[1] = bitcast<vec2<u64>>(a % b);
        output[2] = bitcast<vec2<u64>>(a << counts);
        output[3] = bitcast<vec2<u64>>(a >> counts);
        output[4] = bitcast<vec2<u64>>(-a);
        output[5] = bitcast<vec2<u64>>(~a);
        output[6] = x / y;
        output[7] = x % y;
        var shifted = x;
        shifted <<= counts;
        output[8] = shifted;
        output[9] = x >> counts;
        output[10] = ~x;
        var signed_right = a.y;
        signed_right >>= counts.y;
        output[11] = vec2<u64>(bitcast<u64>(a.x / b.x), bitcast<u64>(signed_right));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func int64ArithmeticLowersScalarAndVectorEdges(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(int64ArithmeticShader), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [80, 192])
    guard case .text(_, let text) = shader.artifact else { Issue.record("Expected source"); return }
    #expect(text.contains("saga_i64_div"))
    #expect(text.contains("saga_u64_rem"))
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.shaderInt64))
    } else {
        #expect(text.contains("& 63u"))
    }
}

@Test(arguments: [
    "a >> 64u",
    "a << 1lu",
    "vec2<i64>(a) >> vec2u(0u, 64u)",
    "a / 0li",
    "-9223372036854775808li % -1li",
    "vec2<i64>(-9223372036854775808li) / vec2<i64>(-1li, 2li)",
    "0x8000000000000000lu << 1u",
])
func int64ArithmeticRejectsStaticFaults(_ expression: String) {
    let source = """
        @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) lane: u32) {
            let a = i64(lane);
            _ = \(expression);
        }
        """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.shaderInt64))
    }
}

@Test func int64ArithmeticValidatesPipelineOperandsAt64Bits() throws {
    func shader(dividend: String) -> ShaderInput {
        .wgsl("""
            override shift: u32 = 63u;
            override divisor: i32 = 1i;
            @group(0) @binding(0) var<storage> input: i64;
            @group(0) @binding(1) var<storage, read_write> output: array<i64, 2>;
            @compute @workgroup_size(1) fn main() {
                output[0] = input >> shift;
                output[1] = \(dividend) / i64(divisor);
            }
            """)
    }
    let compiler = IndependentShaderCompiler()
    let source = shader(dividend: "input")
    _ = try compiler.compileCompute(source, to: .msl)
    for constants: [String: Double] in [["shift": 64], ["divisor": 0]] {
        #expect(throws: ShaderFailure.self) {
            try compiler.compileCompute(source, to: .msl, constants: constants)
        }
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(shader(dividend: "-9223372036854775808li"), to: .msl, constants: ["divisor": -1])
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func int64ArithmeticMatchesCPUOnMetal(zeroDivisor: Bool) async throws {
    let a: [Int64] = [.min, -91]
    let b: [Int64] = [zeroDivisor ? 0 : -1, 7]
    let x: [UInt64] = [.max, 0x8000_0000_0000_0001]
    let y: [UInt64] = [0, 0x4000_0000_0000_0001]
    let counts: [UInt32] = zeroDivisor ? [65, 32] : [64, 127]
    let shifts = counts.map { $0 & 63 }
    let effectiveDivisors = zip(a, b).map { value, divisor in
        divisor == 0 || (value == .min && divisor == -1) ? 1 : divisor
    }
    let signedQuotients = zip(a, effectiveDivisors).map { UInt64(bitPattern: $0 / $1) }
    let signedRemainders = zip(a, effectiveDivisors).map { UInt64(bitPattern: $0 % $1) }
    let signedRight = zip(a, shifts).map { UInt64(bitPattern: $0 >> $1) }
    let expectedGroups: [[UInt64]] = [
        signedQuotients, signedRemainders,
        zip(a, shifts).map { UInt64(bitPattern: $0 << $1) }, signedRight,
        a.map { UInt64(bitPattern: 0 &- $0) }, a.map { UInt64(bitPattern: ~$0) },
        zip(x, y).map { $0 / max(1, $1) }, zip(x, y).map { $0 % max(1, $1) },
        zip(x, shifts).map { $0 << $1 }, zip(x, shifts).map { $0 >> $1 }, x.map { ~$0 },
        [signedQuotients[0], signedRight[1]],
    ]
    func words(_ values: [UInt64]) -> [UInt32] {
        values.flatMap { [UInt32(truncatingIfNeeded: $0), UInt32($0 >> 32)] }
    }
    let inputWords = words((a + b).map(UInt64.init(bitPattern:)) + x + y) + counts + [0, 0]
    let expected = words(expectedGroups.flatMap { $0 })
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(int64ArithmeticShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 80, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 192, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 192, usage: [.mapRead, .copyDestination]))
    let actual = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: inputWords),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 192), .read(buffer: readback, byteCount: 192),
    ]).wait()
    #expect(actual == [expected])
    await device.close()
}
#endif
