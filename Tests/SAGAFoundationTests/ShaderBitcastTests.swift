import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let bitcastShader = """
    alias Words = vec4<u32>;
    const FLOATS = bitcast<vec4f>(vec4u(0x3f800000u, 0xc0200000u, 0x80000000u, 0x3f000000u));
    const PACKED: Words = bitcast<Words>(FLOATS);
    const_assert bitcast<u32>(4294967295) == 0xffffffffu;
    const_assert bitcast<vec2u>(vec2(1, 4294967295)).y == 0xffffffffu;
    const_assert bitcast<u32>(bitcast<f32>(-2147483648)) == 0x80000000u;
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 1>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 5>;
    @compute @workgroup_size(1) fn run() {
        let words = input[0];
        let floats = bitcast<vec4<f32>>(words);
        output[0] = bitcast<vec4u>(floats);
        output[1] = bitcast<vec4u>(bitcast<vec4i>(floats));
        output[2] = bitcast<vec4u>(bitcast<vec4f>(bitcast<vec4i>(words)));
        let pairs = bitcast<vec2u>(bitcast<vec2f>(words.xy));
        let triple = bitcast<vec3u>(bitcast<vec3f>(words.yzw));
        output[3] = vec4u(pairs, triple.z, bitcast<u32>(floats.x + 1.0));
        const high = bitcast<u32>(-1i);
        output[4] = vec4u(PACKED.xy, bitcast<u32>(-0.0f), high);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsScalarAndVectorBitcasts(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(bitcastShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("bitcast<"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains("as_type<float4>(") && source.contains("as_type<uint3>("))
    case .hlsl:
        #expect(source.contains("asfloat(") && source.contains("asint(") && source.contains("asuint("))
    case .glsl:
        #expect(source.contains("uintBitsToFloat(") && source.contains("floatBitsToInt("))
        #expect(source.contains("intBitsToFloat(") && source.contains("floatBitsToUint("))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    ("const X = bitcast<u32>(-1);", ShaderDiagnosticCode.invalidType),
    ("const X = bitcast<f32>(0x7f800000u);", .invalidType),
    ("const X = bitcast<vec2f>(vec2u(0u, 0x7fc00000u));", .invalidType),
    ("const X = bitcast<vec2u>(1u);", .typeMismatch),
    ("const X = bitcast<bool>(1u);", .typeMismatch),
    ("alias vec2 = vec2u; const X = bitcast<vec2<u32>>(vec2u(1));", .unexpectedToken),
    ("const X = bitcast<u32>(1i, 2i);", .typeMismatch),
    ("@compute @workgroup_size(1) fn run() { let x = bitcast<f32>(0x7f800000u); }", .invalidType),
    ("@compute @workgroup_size(1) fn run() { let x = bitcast<vec3u>(vec2f(1)); }", .typeMismatch),
])
func rejectsInvalidBitcasts(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid bitcast accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

#if canImport(Metal)
@Test func preservesScalarAndVectorBitPatternsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(bitcastShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 80, usage: [.mapRead, .copyDestination]))
        let words: [UInt32] = [0x3f800000, 0xc0200000, 0x3f000000, 0x00800000]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: words),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        ])
        let expected = words + words + words + [words[0], words[1], words[3], 0x40000000] +
            [0x3f800000, 0xc0200000, 0x80000000, 0xffffffff]
        #expect(try await ticket.wait() == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
