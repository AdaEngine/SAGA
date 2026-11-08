import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let quantizeShader = """
    const_assert quantizeToF16(65504) == 65504.0f;
    const_assert quantizeToF16(0x1.002p0f) == 1.0f;
    const_assert quantizeToF16(0x1.006p0f) == 0x1.008p0f;
    const_assert quantizeToF16(0x1p-24f) == 0x1p-24f;
    const_assert bitcast<u32>(quantizeToF16(-0x1p-26f)) == 0x80000000u;
    const_assert bitcast<u32>(quantizeToF16(-0.0)) == 0x80000000u;
    const_assert quantizeToF16(vec2(0.5, -1.0)).y == -1.0f;
    const_assert quantizeToF16(vec4f(0.125, 0.25, 0.5, 1)).z == 0.5f;
    @group(0) @binding(0) var<storage, read> input: array<vec4f, 3>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 3>;
    @compute @workgroup_size(1) fn run() {
        let b = input[1];
        let c = input[2];
        output[0] = bitcast<vec4u>(quantizeToF16(input[0]));
        output[1] = bitcast<vec4u>(vec4f(quantizeToF16(b.xy), quantizeToF16(b.z), quantizeToF16(b.w)));
        output[2] = bitcast<vec4u>(vec4f(quantizeToF16(c.xyz), quantizeToF16(c.w)));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func quantizeToF16UsesCoreF32Types(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(.wgsl(quantizeShader), capabilities: [])
    let shader = try compiler.compileCompute(.wgsl(quantizeShader), to: format)
    #expect(!shader.requiresShaderF16)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("quantizeToF16(") && !source.contains("enable f16;"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [])
    case .msl:
        #expect(source.contains("half4(") && source.contains("half3("))
    case .hlsl:
        #expect(source.contains("f16tof32(f32tof16("))
    case .glsl:
        #expect(source.contains("unpackHalf2x16(packHalf2x16("))
        #expect(!source.contains("GL_EXT_shader_explicit_arithmetic_types_float16"))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    ("const x = quantizeToF16(1u);", ShaderDiagnosticCode.typeMismatch),
    ("const x = quantizeToF16(1i);", .typeMismatch),
    ("enable f16; const x = quantizeToF16(vec2h(1));", .typeMismatch),
    ("const x = quantizeToF16(1.0f, 2.0f);", .typeMismatch),
    ("const x = quantizeToF16(65505);", .invalidType),
    ("const x = quantizeToF16(vec3f(0, -65505, 1));", .invalidType),
    ("const x = true || quantizeToF16(70000f) > 0.0f;", .invalidType),
    ("@compute @workgroup_size(1) fn run() { let x = quantizeToF16(65505f); }", .invalidType),
])
func diagnosesInvalidQuantizeToF16Operands(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
        Issue.record("Invalid quantizeToF16 accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

#if canImport(Metal)
@Test func quantizeToF16RoundsScalarsAndVectorsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(quantizeShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 48, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
        let words: [UInt32] = [
            0x3f800800, 0x3f803800, 0x80000000, 0x477fe000,
            0x38800000, 0xb2800000, 0x33800000, 0xbf802000,
            0x402001a3, 0x3eaaaaab, 0xba83126f, 0xc77fe000,
        ]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: words),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
        ])
        var actual = try #require(try await ticket.wait().first)
        try #require(actual.count == 12)
        // WGSL also permits flushing the positive half subnormal to zero.
        if actual[6] == 0 { actual[6] = 0x33800000 }
        #expect(actual == [
            0x3f800000, 0x3f804000, 0x80000000, 0x477fe000,
            0x38800000, 0x80000000, 0x33800000, 0xbf802000,
            0x40200000, 0x3eaaa000, 0xba832000, 0xc77fe000,
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
