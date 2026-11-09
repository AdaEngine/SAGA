import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvInt16ArithmeticInput = ShaderInput.wgsl("""
    enable wgpu_int16;
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 6>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 10>;

    @compute @workgroup_size(1) fn run() {
        let bits = vec4<u16>(input[0]);
        let signedBits = vec4<i16>(input[0]);
        let counts = input[1];
        output[0] = vec4u(bits << counts);
        output[1] = vec4u(bits >> counts);
        output[2] = bitcast<vec4u>(vec4i(signedBits << counts));
        output[3] = bitcast<vec4u>(vec4i(signedBits >> counts));

        let signed = vec4<i16>(input[2]);
        let signedDivisor = vec4<i16>(input[3]);
        let unsigned = vec4<u16>(input[4]);
        let unsignedDivisor = vec4<u16>(input[5]);
        output[4] = bitcast<vec4u>(vec4i(signed / signedDivisor));
        output[5] = bitcast<vec4u>(vec4i(signed % signedDivisor));
        output[6] = vec4u(unsigned / unsignedDivisor);
        output[7] = vec4u(unsigned % unsignedDivisor);
        output[8] = vec4u(u32(bits.x << counts.x), u32(bits.y >> counts.y),
                          bitcast<u32>(i32(signedBits.z << counts.w)), bitcast<u32>(i32(signedBits.w >> counts.z)));
        output[9] = vec4u(bitcast<u32>(i32(signed.x / signedDivisor.x)), bitcast<u32>(i32(signed.w % signedDivisor.w)),
                          u32(unsigned.y / unsignedDivisor.y), u32(unsigned.y % unsignedDivisor.y));
    }
    """)

private func exportInt16Arithmetic(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvInt16ArithmeticInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvInt16ArithmeticUsesBoundedCountsAndSafeDivisors() throws {
    let words = try exportInt16Arithmetic(IndependentShaderCompiler())
    var maskedCounts: Set<UInt32> = []
    var guardedDivisors: Set<UInt32> = []
    var shiftCounts: [UInt32] = []
    var divisors: [UInt32] = []
    var opcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        switch opcode {
        case 134 where count == 5, 137 where count == 5:
            divisors.append(words[offset + 4])
        case 169 where count == 6: guardedDivisors.insert(words[offset + 2]) // OpSelect
        case 194...196 where count == 5: shiftCounts.append(words[offset + 4])
        case 199 where count == 5: maskedCounts.insert(words[offset + 2]) // OpBitwiseAnd
        default: break
        }
        offset += count
    }
    #expect(opcodes.isSuperset(of: [134, 137, 194, 195, 196]))
    #expect(opcodes.isDisjoint(with: [135, 138, 139])) // Signed division/remainder use unsigned magnitudes.
    #expect(shiftCounts.count == 8 && shiftCounts.allSatisfy(maskedCounts.contains))
    #expect(divisors.count >= 8 && divisors.allSatisfy(guardedDivisors.contains))
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvInt16ArithmeticRoundTripsNativeShaderFormats(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportInt16Arithmetic(compiler)), to: format)
    #expect(shader.requiresShaderInt16)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [96, 160])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.shaderInt16))
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvInt16ArithmeticPreservesRuntimeEdgeResultsOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shaderInput = try roundTrip ? ShaderInput.spirv(exportInt16Arithmetic(compiler)) : spirvInt16ArithmeticInput
    let shader = try compiler.compileCompute(shaderInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 96, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 160, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 160, usage: [.mapRead, .copyDestination]))
        let seeds: [UInt32] = [
            0x8001, 0x8003, 0xffff, 0x8000, 16, 17, 15, .max,
            0x8000, 0xfff9, 7, 0xffed, 0xffff, 0, 0xfffd, 4,
            0xffff, 9, 17, 0x8000, 2, 0, 5, 3,
        ]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: seeds),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 160), .read(buffer: readback, byteCount: 160),
        ]).wait()
        let expected: [UInt32] = [
            0x8001, 6, 0x8000, 0, 0x8001, 0x4001, 1, 1,
            0xffff8001, 6, 0xffff8000, 0, 0xffff8001, 0xffffc001, .max, .max,
            0xffff8000, 0xfffffff9, 0xfffffffe, 0xfffffffc, 0, 0, 1, 0xfffffffd,
            0x7fff, 9, 3, 10922, 1, 0, 2, 2,
            0x8001, 0x4001, 0xffff8000, .max, 0xffff8000, 0xfffffffd, 9, 0,
        ]
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
