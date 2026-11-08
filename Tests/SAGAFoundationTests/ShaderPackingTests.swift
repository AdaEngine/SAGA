import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let packingShader = """
    const packed = vec4u(pack4x8snorm(vec4f(-2, -0.5, 0.5, 2)), pack4x8unorm(vec4f(-2, -0.5, 0.5, 2)),
                        pack2x16snorm(vec2f(-0.5, 0.5)), pack2x16float(vec2f(1, -2)));
    const_assert unpack4xI8(0x7f00ff80u).x == -128i;
    const_assert unpack4xU8(0x7f00ff80u).w == 127u;
    const_assert unpack2x16float(pack2x16float(vec2f(1.00048828125, 5.960464477539063e-8))).x == 1.0f;
    const_assert pack2x16float(vec2f(5.960464477539063e-8, -0.0)) == 0x80000001u;
    const_assert unpack2x16float(1u).x == 5.960464477539063e-8f;
    const_assert pack2x16float(vec2f(65504, -65504)) == 0xfbff7bffu;
    const_assert pack4xI8Clamp(vec4(-129, -1, 127, 128)) == 0x7f7fff80u;
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 5>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 9>;
    @compute @workgroup_size(1) fn run() {
        let normalized = bitcast<vec4f>(input[0]);
        let halfValues = bitcast<vec2f>(input[1].xy);
        let word = input[1].z;
        let signed = bitcast<vec4i>(input[2]);
        let unsigned = input[3];
        let channels = input[4];
        output[0] = vec4u(pack4x8snorm(normalized), pack4x8unorm(normalized),
                          pack2x16snorm(normalized.yz), pack2x16unorm(normalized.zw));
        output[1] = vec4u(pack4xI8(signed), pack4xI8Clamp(signed), pack4xU8(unsigned), pack4xU8Clamp(unsigned));
        output[2] = bitcast<vec4u>(unpack4xI8(word));
        output[3] = unpack4xU8(word);
        output[4] = bitcast<vec4u>(unpack4x8snorm(channels.x));
        output[5] = bitcast<vec4u>(unpack4x8unorm(channels.y));
        output[6] = vec4u(bitcast<vec2u>(unpack2x16snorm(channels.z)), bitcast<vec2u>(unpack2x16unorm(channels.w)));
        output[7] = vec4u(pack2x16float(halfValues), bitcast<vec2u>(unpack2x16float(input[1].w)), pack2x16float(unpack2x16float(input[1].w)));
        output[8] = packed;
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsNormalizedHalfAndIntegerPacking(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(packingShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("pack2x16float(") && source.contains("unpack4xI8("))
        _ = try compiler.parseAndValidate(.wgsl(source))
        for target: ShaderOutputFormat in [.msl, .hlsl, .glsl] {
            _ = try compiler.compileCompute(.wgsl(source), to: target)
        }
    case .msl: #expect(source.contains("as_type<half2>(") && source.contains("floor("))
    case .hlsl: #expect(source.contains("f32tof16(") && source.contains("f16tof32(") && source.contains("floor("))
    case .glsl: #expect(source.contains("packHalf2x16(") && source.contains("unpackHalf2x16(") && source.contains("floor("))
    default: Issue.record("Unexpected shader target")
    }
}

@Test(arguments: [
    ("const x = pack2x16float(vec2f(65505, 1));", ShaderDiagnosticCode.invalidType),
    ("const x = unpack2x16float(0x7c00u);", .invalidType),
    ("const x = pack4x8snorm(vec2f(0));", .typeMismatch),
    ("const x = pack4xI8(vec4u(0));", .typeMismatch),
    ("const x = unpack4xU8(1i);", .typeMismatch),
    ("@compute @workgroup_size(1) fn run() { let x = pack2x16float(vec2f(65505, 1)); }", .invalidType),
    ("@compute @workgroup_size(1) fn run() { let x = pack4x8unorm(1.0f); }", .typeMismatch),
])
func rejectsInvalidPackingExpressions(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid packing builtin accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

private let packingInputWords: [UInt32] = [
    0xc0000000, 0xbf000000, 0x3f000000, 0x40000000,
    0x3f800000, 0xc0000000, 0x7f00ff80, 0xc0003c00,
    0xffffff7f, 0xffffffff, 127, 128, 256, 255, 511, 0x12345678,
    0x7f008081, 0xff00ff00, 0x7fff8000, 0xffff0000,
]

private let expectedPackingWords: [UInt32] = [
    0x7f40c181, 0xff800000, 0x4000c001, 0xffff8000,
    0x807fff7f, 0x7f7fff80, 0x78ffff00, 0xffffffff,
    0xffffff80, 0xffffffff, 0, 127, 128, 255, 0, 127,
    0xbf800000, 0xbf800000, 0, 0x3f800000, 0, 0x3f800000, 0, 0x3f800000,
    0xbf800000, 0x3f800000, 0, 0x3f800000,
    0xc0003c00, 0x3f800000, 0xc0000000, 0xc0003c00,
    0x7f40c181, 0xff800000, 0x4000c001, 0xc0003c00,
]

#if canImport(Metal)
@Test(arguments: [false, true])
func packsAndUnpacksWordsOnMetal(roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let source: String
    if roundTrip {
        let artifact = try compiler.compileCompute(.wgsl(packingShader), to: .wgsl).artifact
        guard case .text(.wgsl, let generated) = artifact else { Issue.record("Expected WGSL source"); return }
        source = generated
    } else {
        source = packingShader
    }
    let shader = try compiler.compileCompute(.wgsl(source), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 80, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 144, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 144, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: packingInputWords),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 144), .read(buffer: read, byteCount: 144),
        ])
        #expect(try await ticket.wait() == [expectedPackingWords])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
