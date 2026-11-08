import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let halfBitcastShader = """
    enable f16;
    alias HalfPair = vec2h;
    const PAIR = bitcast<HalfPair>(0xc0003c00u);
    const_assert PAIR.x == 1h && PAIR.y == -2h;
    const_assert bitcast<u32>(vec2h(-0.0h, 0x1p-24h)) == 0x00018000u;
    const_assert bitcast<vec2u>(bitcast<vec4h>(vec2u(0xc0003c00u, 0x04008000u))).y == 0x04008000u;
    const_assert bitcast<u32>(bitcast<vec2h>(1.0)) == 0x3f800000u;
    const_assert bitcast<u32>(vec2(1, 2)) == 0x40003c00u;
    const_assert bitcast<f16>(0.5) == 0.5h;
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 1>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 5>;
    struct UniformBits { basis: mat2x2h, halfWords: vec4h }
    @group(0) @binding(2) var<uniform> settings: UniformBits;
    @compute @workgroup_size(1) fn run() {
        let words = input[0];
        let halves = bitcast<vec4h>(words.xy);
        let fromSigned = bitcast<vec4h>(bitcast<vec2i>(words.zw));
        let fromFloats = bitcast<vec4h>(bitcast<vec2f>(words.zw));
        output[0] = vec4u(bitcast<vec2u>(halves), bitcast<vec2u>(fromSigned));
        output[1] = vec4u(bitcast<u32>(bitcast<vec2h>(words.x)), bitcast<u32>(bitcast<i32>(halves.xy)),
                          bitcast<u32>(bitcast<f32>(fromFloats.xy)), bitcast<u32>(bitcast<vec2h>(bitcast<f32>(words.w))));
        output[2] = vec4u(bitcast<vec2u>(bitcast<vec2f>(fromFloats)), bitcast<vec2u>(bitcast<vec2i>(halves)));
        let same = bitcast<vec3h>(halves.xyz);
        output[3] = vec4u(bitcast<u32>(vec2h(same.x, bitcast<f16>(same.y))), bitcast<u32>(halves.zw),
                          bitcast<u32>(PAIR), bitcast<u32>(vec2h(PAIR.y, PAIR.x)));
        output[4] = vec4u(bitcast<vec2u>(settings.halfWords), bitcast<u32>(settings.basis[0]), bitcast<u32>(settings.basis[1]));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsHalfPrecisionBitcasts(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(halfBitcastShader), to: format)
    #expect(shader.requiresShaderF16)
    #expect(shader.uniformBindings.map(\.minimumByteSize) == [16])
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("bitcast<vec4<f16>>"))
        _ = try compiler.parseAndValidate(.wgsl(body), capabilities: [.shaderFloat16])
    case .msl: #expect(body.contains("as_type<half4>(") && body.contains("as_type<uint2>("))
    case .hlsl:
        #expect(body.contains("asfloat16(") && body.contains("asuint16("))
        #expect(!body.contains("f16tof32("))
    case .glsl:
        #expect(body.contains("unpackFloat2x16(") && body.contains("packFloat2x16("))
        #expect(!body.contains("unpackHalf2x16("))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    ("const x = bitcast<vec2h>(0u);", ShaderDiagnosticCode.unsupportedCapability),
    ("enable f16; const x = bitcast<u32>(1h);", .typeMismatch),
    ("enable f16; const x = bitcast<f16>(1.0f);", .typeMismatch),
    ("enable f16; const x = bitcast<vec2u>(vec3h(1));", .typeMismatch),
    ("enable f16; const x = bitcast<vec4h>(vec4u(1));", .typeMismatch),
    ("enable f16; const x = bitcast<vec2h>(0x7c003c00u);", .invalidType),
    ("enable f16; const x = bitcast<vec4h>(vec2u(0u, 0x7e000000u));", .invalidType),
    ("enable f16; @compute @workgroup_size(1) fn run() { let x = bitcast<vec2h>(0x00007c00u); }", .invalidType),
])
func rejectsInvalidHalfPrecisionBitcasts(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
        Issue.record("Invalid half bitcast accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

#if canImport(Metal)
@Test func preservesPackedHalfBitPatternsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfBitcastShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let settings = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 80, usage: [.mapRead, .copyDestination]))
        let words: [UInt32] = [0xc0003c00, 0x04008000, 0xbc004400, 0x3c003800]
        let uniformWords: [UInt32] = [0x00018000, 0x04003c00, 0x80000001, 0x4400c000]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: words),
            .write(buffer: settings, offset: 0, words: uniformWords),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output,
                             .init(binding: 2): settings],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        ])
        let expected = words + [words[0], words[0], words[2], words[3]]
            + [words[2], words[3], words[0], words[1]] + [words[0], words[1], words[0], 0x3c00c000]
            + [uniformWords[2], uniformWords[3], uniformWords[0], uniformWords[1]]
        #expect(try await ticket.wait() == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
