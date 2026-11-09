import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvIntegerBitsInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read> inputs: array<vec4u, 3>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 16>;
    @compute @workgroup_size(1) fn run() {
        let bits = inputs[0];
        let signed = bitcast<vec4i>(bits);
        let shifts = inputs[1];
        let ranges = inputs[2];
        output[0] = bits << shifts;
        output[1] = bits >> shifts;
        output[2] = bitcast<vec4u>(signed << shifts);
        output[3] = bitcast<vec4u>(signed >> shifts);
        output[4] = vec4u(bits.w << shifts.y, bits.z >> shifts.z,
                          bitcast<u32>(signed.w << shifts.w), bitcast<u32>(signed.z >> shifts.w));
        output[5] = countOneBits(bits);
        output[6] = bitcast<vec4u>(reverseBits(signed));
        output[7] = countLeadingZeros(bits);
        output[8] = bitcast<vec4u>(countTrailingZeros(signed));
        output[9] = firstLeadingBit(bits);
        output[10] = bitcast<vec4u>(firstLeadingBit(signed));
        output[11] = bitcast<vec4u>(firstTrailingBit(signed));
        output[12] = extractBits(bits, ranges.x, ranges.y);
        output[13] = bitcast<vec4u>(extractBits(signed, ranges.x, ranges.y));
        output[14] = insertBits(bits, vec4u(5u), ranges.x, ranges.y);
        output[15] = vec4u(extractBits(bits.w, ranges.z, ranges.y), insertBits(bits.y, 7u, ranges.z, ranges.y),
                           extractBits(bits.z, 0u, ranges.w), insertBits(bits.x, 7u, 0u, ranges.w));
    }
    """)

private func exportIntegerBits(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvIntegerBitsInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvIntegerBitsCodeGeneratorRoundTripsIntegerBitOperations(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportIntegerBits(compiler)), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [48, 256])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

#if canImport(Metal)
@Test func spirvIntegerBitsCodeGeneratorPreservesShiftAndBitfieldEdgesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportIntegerBits(compiler)), to: .msl)
    let inputWords: [UInt32] = [0, 1, 0x80000000, .max, 0, 32, 33, .max, 28, .max, .max, 32]
    let expected: [UInt32] = [
        0, 1, 0, 0x80000000, 0, 1, 0x40000000, 1,
        0, 1, 0, 0x80000000, 0, 1, 0xc0000000, .max,
        .max, 0x40000000, 0x80000000, .max,
        0, 1, 1, 32, 0, 0x80000000, 1, .max, 32, 31, 0, 0, 32, 0, 31, 0,
        .max, 0, 31, 31, .max, 0, 30, .max, .max, 0, 31, 0,
        0, 0, 8, 15, 0, 0, 0xfffffff8, .max, 0x50000000, 0x50000001, 0x50000000, 0x5fffffff,
        0, 1, 0x80000000, 7,
    ]
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 48, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 256, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 256, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: inputWords),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 256), .read(buffer: readback, byteCount: 256),
        ]).wait()
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
