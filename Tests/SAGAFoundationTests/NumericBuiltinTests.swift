import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let numericShader = """
    const_assert abs(vec2i(-2147483647i - 1i, -7i)).x == bitcast<i32>(0x80000000u);
    const_assert abs(vec2u(0xffffffffu, 1u)).x == 0xffffffffu;
    const_assert max(vec2(4294967295, 1), vec2u(0)).x == 0xffffffffu;
    const_assert round(vec4f(-2.5, -1.5, 1.5, 2.5)).w == 2.0f;
    const_assert fract(-1.25f) == 0.75f;
    const_assert sign(vec2(-4, 0)).x == -1;
    const_assert clamp(vec2f(-4, 5), vec2f(-1, 0), vec2f(2, 3)).y == 3.0f;
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 6>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 13>;
    @compute @workgroup_size(1) fn run() {
        let signed = bitcast<vec4i>(input[0]);
        let unsigned = input[1];
        let values = bitcast<vec4f>(input[2]);
        let scales = bitcast<vec4f>(input[5]);
        output[0] = bitcast<vec4u>(abs(signed));
        output[1] = bitcast<vec4u>(sign(signed));
        output[2] = min(unsigned, vec4(3));
        output[3] = max(unsigned, vec4(3));
        output[4] = clamp(unsigned, input[3], input[4]);
        output[5] = bitcast<vec4u>(round(values));
        output[6] = bitcast<vec4u>(floor(values));
        output[7] = bitcast<vec4u>(ceil(values));
        output[8] = bitcast<vec4u>(trunc(values));
        output[9] = bitcast<vec4u>(fract(values));
        output[10] = bitcast<vec4u>(saturate(values));
        output[11] = bitcast<vec4u>(vec4f(sqrt(scales.x), inverseSqrt(scales.x), sin(scales.y), cos(scales.y)));
        output[12] = bitcast<vec4u>(vec4f(abs(values.x), sign(values.x), min(values.x, values.w), max(values.x, values.w)));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsScalarAndVectorNumericBuiltins(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(numericShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("inverseSqrt(") && source.contains("round("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl: #expect(source.contains("rint(") && source.contains("rsqrt("))
    case .hlsl: #expect(source.contains("frac(") && source.contains("rsqrt("))
    case .glsl: #expect(source.contains("roundEven(") && source.contains("inversesqrt("))
    default: Issue.record("Unexpected shader target")
    }
}

@Test(arguments: [
    ("const x = sign(vec2u(1));", ShaderDiagnosticCode.typeMismatch),
    ("const x = round(vec2i(1));", .typeMismatch),
    ("const x = min(vec2u(1), vec3u(2));", .typeMismatch),
    ("const x = clamp(vec2f(0), vec2f(0, 2), vec2f(1));", .invalidType),
    ("const x = sqrt(vec2f(1, -1));", .invalidType),
    ("@compute @workgroup_size(1) fn run() { var x = vec2i(0); let y = clamp(x, vec2i(2), vec2i(1)); }", .invalidType),
    ("@compute @workgroup_size(1) fn run() { var x = 0i; let y = floor(x); }", .typeMismatch),
])
func rejectsInvalidNumericBuiltins(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid numeric builtin accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

private let numericInput: [UInt32] = [
    0x80000000, 0xfffffff9, 0, 9, 0xffffffff, 1, 8, 2,
    0xc0200000, 0xbfc00000, 0x3fc00000, 0x40200000, 0, 4, 0, 5, 4, 2, 3, 1, 0x40800000, 0, 0, 0,
]
private let expectedNumericIntegers: [UInt32] = [
    0x80000000, 7, 0, 9, 0xffffffff, 0xffffffff, 0, 1,
    3, 1, 3, 2, 0xffffffff, 3, 8, 3, 4, 2, 3, 1,
]
private let expectedNumericFloats: [Float] = [
    -2, -2, 2, 2, -3, -2, 1, 2, -2, -1, 2, 3, -2, -1, 1, 2,
    0.5, 0.5, 0.5, 0.5, 0, 0, 1, 1, 2, 0.5, 0, 1, 2.5, -1, -2.5, 2.5,
]

#if canImport(Metal)
@Test func computesNumericBuiltinsAtIntegerAndRoundingBoundariesOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(numericShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 96, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 208, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 208, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: numericInput),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 208), .read(buffer: read, byteCount: 208),
        ])
        let words = try #require(try await ticket.wait().first)
        #expect(Array(words.prefix(20)) == expectedNumericIntegers)
        #expect(words.dropFirst(20).map { Float(bitPattern: $0) } == expectedNumericFloats)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
