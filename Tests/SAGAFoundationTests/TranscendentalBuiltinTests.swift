import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let transcendentalShader = """
    const_assert exp2(4.0) == 16.0;
    const_assert log2(0.125) == -3.0;
    const_assert log2(4.9406564584124654e-324) == -1074.0;
    const_assert log2(exp2(-1074.0)) == -1074.0;
    const_assert abs(exp2(1023.5) / exp2(1023.0) - 1.4142135623730951) < 1e-14;
    const_assert abs(log(exp(0.5)) - 0.5) < 1e-14;
    const_assert abs(atan2(1.0, -1.0) - 2.356194490192345) < 1e-14;
    const_assert abs(sin(1e300) + 0.8178819121159085) < 1e-14;
    const_assert abs(cos(1e30f) + 0.6116048f) < 2e-6f;
    const_assert abs(tan(1e100) + 0.41162296288324984) < 1e-14;
    const_assert abs(asinh(sinh(-2.0)) + 2.0) < 1e-14;
    const_assert abs(acosh(cosh(2.0)) - 2.0) < 1e-14;
    const_assert abs(atanh(tanh(0.5)) - 0.5) < 1e-14;
    const_assert abs(pow(vec2f(9, 16), vec2f(0.5)).y - 4.0f) < 1e-6f;
    @group(0) @binding(0) var<storage, read> input: array<vec4f, 3>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 6>;
    @compute @workgroup_size(1) fn run() {
        let x = input[0];
        let y = input[1];
        let angle = atan2(x.z, x.w);
        output[0] = vec4f(sin(x.x), cos(x.y), tan(x.x), angle);
        output[1] = vec4f(asin(x.x), acos(x.x), atan(x.z), degrees(angle));
        output[2] = vec4f(sinh(x.x), cosh(x.x), tanh(x.y), radians(y.w));
        output[3] = vec4f(asinh(x).y, acosh(y).x, atanh(x.x), pow(y.z, x.x));
        output[4] = vec4f(exp(x.x), exp2(y.x), log(y.x), log2(y.y));
        output[5] = exp2(input[2]);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsTranscendentalBuiltinsAndAtan2ArgumentOrder(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(transcendentalShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(source.contains("exp2(") && source.contains("log2(") && source.contains("pow("))
    switch format {
    case .wgsl:
        #expect(source.contains("atan2(") && source.contains("asinh("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl: #expect(source.contains("atan2(") && source.contains("acosh("))
    case .hlsl: #expect(source.contains("atan2(") && source.contains("log(") && source.contains("sqrt("))
    case .glsl: #expect(source.contains("atan(") && !source.contains("atan2("))
    default: Issue.record("Unexpected shader target")
    }
}

@Test(arguments: [
    ("const x = log(0.0);", ShaderDiagnosticCode.invalidType),
    ("const x = asin(vec2f(-2, 0));", .invalidType),
    ("const x = acos(1.01f);", .invalidType),
    ("const x = acosh(0.5);", .invalidType),
    ("const x = atanh(1.0);", .invalidType),
    ("const x = exp(1000.0);", .invalidType),
    ("const x = pow(-2.0, 2.0);", .invalidType),
    ("const x = atan2(vec2f(1), 2.0f);", .typeMismatch),
    ("@compute @workgroup_size(1) fn run() { let x = log2(-1.0f); }", .invalidType),
    ("@compute @workgroup_size(1) fn run() { var x = 1u; let y = sin(x); }", .typeMismatch),
])
func diagnosesInvalidTranscendentalInputs(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid transcendental builtin accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

private let transcendentalInput: [Float] = [0.5, -0.5, 1, -1, 2, 8, 9, 180, 1, 2, 3, 4]
private let expectedTranscendental: [Float] = [
    0.47942555, 0.87758255, 0.5463025, 2.3561945,
    0.5235988, 1.0471976, 0.7853982, 135,
    0.5210953, 1.127626, -0.46211717, 3.1415927,
    -0.48121184, 1.316958, 0.54930615, 3,
    1.6487213, 4, 0.6931472, 3, 2, 4, 8, 16,
]

#if canImport(Metal)
@Test func evaluatesTranscendentalBuiltinsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(transcendentalShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 48, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 96, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: transcendentalInput.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
        ])
        let values = try #require(try await ticket.wait().first).map { Float(bitPattern: $0) }
        #expect(values.count == expectedTranscendental.count)
        for (value, expected) in zip(values, expectedTranscendental) {
            #expect(abs(value - expected) <= max(1, abs(expected)) * 0.0004)
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
