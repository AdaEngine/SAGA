import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvNumericInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 7>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 16>;

    @compute @workgroup_size(1) fn run() {
        let signed = bitcast<vec4i>(input[0]);
        let unsigned = input[1];
        let values = bitcast<vec4f>(input[2]);
        let scales = bitcast<vec4f>(input[5]);
        output[0] = bitcast<vec4u>(abs(signed));
        output[1] = bitcast<vec4u>(sign(signed));
        output[2] = min(unsigned, vec4u(3));
        output[3] = max(unsigned, vec4u(3));
        output[4] = clamp(unsigned, input[3], input[4]);
        output[5] = bitcast<vec4u>(round(values));
        output[6] = bitcast<vec4u>(floor(values));
        output[7] = bitcast<vec4u>(ceil(values));
        output[8] = bitcast<vec4u>(trunc(values));
        output[9] = bitcast<vec4u>(fract(values));
        output[10] = bitcast<vec4u>(saturate(values));
        output[11] = bitcast<vec4u>(vec4f(sqrt(scales.x), inverseSqrt(scales.x), sin(scales.y), cos(scales.y)));
        output[12] = bitcast<vec4u>(vec4f(abs(values.x), sign(values.x), min(values.x, values.w), max(values.x, values.w)));
        let multipliers = input[6];
        let lower = bitcast<i32>(input[3].w);
        let upper = bitcast<i32>(input[4].w);
        output[13] = vec4u(bitcast<u32>(dot(signed, bitcast<vec4i>(multipliers))), dot(unsigned, multipliers),
                          bitcast<u32>(clamp(signed.y, lower, upper)), abs(unsigned.x));
        let blend = mix(values.xy, values.zw, values.z - 1.0f);
        output[14] = bitcast<vec4u>(vec4f(blend, smoothstep(values.w, values.x, scales.y), fma(values.y, scales.x, values.w)));
        output[15] = bitcast<vec4u>(vec4f(ldexp(values.z, bitcast<i32>(multipliers.x)), step(scales.y, values.x),
                                        step(values.x, scales.y), mix(values.x, values.w, 0.5f)));
    }
    """)

private func exportSPIRVNumeric(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvNumericInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvNumericCodeGeneratorUsesIntegerOperationsForIntegerDot() throws {
    let words = try exportSPIRVNumeric(IndependentShaderCompiler())
    var opcodes: Set<UInt32> = []
    var extendedOpcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        if opcode == 12 && count >= 5 { extendedOpcodes.insert(words[offset + 4]) }
        offset += count
    }
    #expect(opcodes.contains(12)) // OpExtInst
    #expect(opcodes.isSuperset(of: [128, 132])) // OpIAdd, OpIMul
    #expect(!opcodes.contains(148)) // OpDot requires floating-point operands.
    #expect(!extendedOpcodes.contains(49)) // GLSL SmoothStep is undefined for reversed edges.
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvNumericCodeGeneratorRoundTripsCoreBuiltins(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportSPIRVNumeric(compiler)), to: format)
    #expect(shader.storageBindings.map(\.isReadOnly) == [true, false])
    guard case .text(_, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    switch format {
    case .wgsl:
        #expect(source.contains("round(") && source.contains("inverseSqrt("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl: #expect(source.contains("rint(") && source.contains("rsqrt("))
    case .hlsl: #expect(source.contains("frac(") && source.contains("rsqrt("))
    case .glsl: #expect(source.contains("roundEven(") && source.contains("inversesqrt("))
    default: Issue.record("Unexpected text target")
    }
}

#if canImport(Metal)
@Test func spirvNumericCodeGeneratorPreservesIntegerAndFloatBoundariesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportSPIRVNumeric(compiler)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 112, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 256, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 256, usage: [.mapRead, .copyDestination]))
        let seeds: [UInt32] = [
            0x80000000, 0xfffffff9, 0, 9, .max, 1, 8, 2,
            0xc0200000, 0xbfc00000, 0x3fc00000, 0x40200000,
            0, 4, 0, 5, 4, 2, 3, 1, 0x40800000, 0, 0, 0, 2, 2, 3, 4,
        ]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: seeds),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 256), .read(buffer: readback, byteCount: 256),
        ])
        let words = try #require(try await ticket.wait().first)
        let integers: [UInt32] = [
            0x80000000, 7, 0, 9, .max, .max, 0, 1,
            3, 1, 3, 2, .max, 3, 8, 3, 4, 2, 3, 1,
        ]
        let floats: [Float] = [
            -2, -2, 2, 2, -3, -2, 1, 2, -2, -1, 2, 3, -2, -1, 1, 2,
            0.5, 0.5, 0.5, 0.5, 0, 0, 1, 1, 2, 0.5, 0, 1, 2.5, -1, -2.5, 2.5,
        ]
        let interpolation: [Float] = [-0.5, 0.5, 0.5, -3.5, 6, 0, 1, 0]
        #expect(words == integers + floats.map(\.bitPattern) + [22, 32, 1, .max] + interpolation.map(\.bitPattern))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
