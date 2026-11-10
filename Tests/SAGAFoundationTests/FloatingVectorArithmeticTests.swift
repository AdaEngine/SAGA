import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let floatingVectorShader = """
    enable f16;
    const_assert (vec2(1.5, 2.5) + 0.5).x == 2.0;
    const_assert (vec2h(-5.5, 7.5) % 2h).x == -1.5h;
    @group(0) @binding(0) var<storage, read> input: array<vec4f, 2>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 8>;
    @compute @workgroup_size(1) fn run() {
        let a = input[0];
        let b = input[1];
        var adjusted = -a;
        adjusted += 2.0;
        adjusted *= 0.5;
        output[0] = quantizeToF16(adjusted);
        output[1] = a / b;
        output[2] = a % b;
        let mask = a < b;
        output[3] = vec4f(a.x / b.x, a.y % b.y, select(0f, 1f, mask.x), select(0f, 1f, mask.y));
        let ah = vec4h(a);
        let bh = vec4h(b);
        output[4] = vec4f((ah + 2.0) * bh);
        output[5] = vec4f((ah % 2h) / (1h + bh));
        output[6] = vec4f(select(vec4h(0), vec4h(1), ah < bh));
        output[7] = 2.0 - a;
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsFloatingVectorArithmetic(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(floatingVectorShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains(" % ") && source.contains("quantizeToF16("))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
    case .msl, .hlsl:
        #expect(source.contains("fmod("))
    case .glsl:
        #expect(source.contains("lessThan(") && source.contains("trunc("))
        // The frontend broadcasts the scalar in `ah % 2h` before code generation.
        #expect(source.split(separator: "\n").contains { line in
            line.contains("f16vec4 saga_remainder_") && line.contains("_rhs = f16vec4(")
        })
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    ("var a = vec2f(1); let x = a + vec3f(1);", ShaderDiagnosticCode.typeMismatch),
    ("var a = vec2h(1); let x = a * vec2f(1);", .typeMismatch),
    ("var a = vec2f(1); let x = a < 2.0f;", .typeMismatch),
    ("var a = vec2f(1); let x = a & vec2f(1);", .typeMismatch),
    ("let x = vec2f(1) / vec2f(1, 0);", .invalidType),
    ("let x = 1.0f % 0.0f;", .invalidType),
])
func rejectsInvalidFloatingVectorArithmetic(body: String, code: ShaderDiagnosticCode) throws {
    let source = "enable f16; @compute @workgroup_size(1) fn run() { \(body) }"
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
        Issue.record("Invalid floating arithmetic accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

#if canImport(Metal)
@Test func evaluatesFloatingVectorArithmeticOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(floatingVectorShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 128, usage: [.mapRead, .copyDestination]))
        let values: [Float] = [-5.5, 7.5, -4, 8, 2, 2, 2, 4]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: values.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
        ])
        let actual = try #require(try await ticket.wait().first).map { Float(bitPattern: $0) }
        #expect(actual == [
            3.75, -2.75, 3, -3, -2.75, 3.75, -2, 2, -1.5, 1.5, 0, 0, -2.75, 1.5, 1, 0,
            -7, 19, -4, 40, -0.5, 0.5, 0, 0, 1, 0, 1, 0, 7.5, -5.5, 6, -6,
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

@Test func floatingArithmeticKeepsLetOperandsAtRuntime() throws {
    let source = """
        @compute @workgroup_size(1) fn run() {
            let vector = vec2f(1);
            let zero = 0.0f;
            let quotient = vector / zero;
            let scalar = 1.0f;
            let remainder = scalar % 0.0f;
        }
        """
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
}
