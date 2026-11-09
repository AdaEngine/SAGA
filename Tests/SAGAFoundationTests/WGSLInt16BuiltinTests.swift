import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let wgslInt16BuiltinInput = ShaderInput.wgsl("""
    enable wgpu_int16;
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 3>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 14>;

    @compute @workgroup_size(1) fn run() {
        let bits = vec4<u16>(input[0]);
        let signed = vec4<i16>(input[0]);
        let ranges = input[1];
        let weights = vec4<u16>(input[2]);
        let signedWeights = vec4<i16>(input[2]);
        output[0] = vec4u(countLeadingZeros(bits));
        output[1] = vec4u(countTrailingZeros(signed));
        output[2] = vec4u(countOneBits(bits));
        output[3] = bitcast<vec4u>(vec4i(reverseBits(signed)));
        output[4] = vec4u(firstLeadingBit(bits));
        output[5] = bitcast<vec4u>(vec4i(firstLeadingBit(signed)));
        output[6] = bitcast<vec4u>(vec4i(firstTrailingBit(signed)));
        output[7] = vec4u(extractBits(bits, ranges.x, ranges.y));
        output[8] = bitcast<vec4u>(vec4i(extractBits(signed, ranges.x, ranges.y)));
        output[9] = vec4u(insertBits(bits, vec4<u16>(u16(0xabc)), ranges.x, ranges.y));
        output[10] = bitcast<vec4u>(vec4i(extractBits(signed, ranges.w, ranges.y)));
        output[11] = bitcast<vec4u>(vec4i(insertBits(signed, vec4<i16>(i16(3)), ranges.z, ranges.y)));
        output[12] = vec4u(u32(extractBits(bits.w, ranges.z, ranges.y)), u32(firstTrailingBit(bits.x)),
                           u32(countLeadingZeros(signed.x)), u32(insertBits(bits.w, u16(0), ranges.x, ranges.w)));
        output[13] = vec4u(bitcast<u32>(i32(dot(signed.zw, signedWeights.xy))), u32(dot(bits.zw, weights.zw)),
                           bitcast<u32>(i32(dot(signed.yz, signedWeights.xy))), u32(countOneBits(bits.w)));
    }
    """)

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func wgslInt16BuiltinWritersPreserveWidthAndLayout(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(wgslInt16BuiltinInput, to: format)
    #expect(shader.requiresShaderInt16)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [48, 224])
    guard case .text(_, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    if format == .wgsl {
        #expect(source.contains("enable wgpu_int16;") && source.contains("vec4<i16>"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.shaderInt16))
    }
    if format == .hlsl { #expect(shader.hlslProfile == "cs_6_2") }
}

@Test func wgslInt16BuiltinsFoldWidthSpecificConstantResults() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_int16;
        const_assert countLeadingZeros(u16(0)) == u16(16);
        const_assert firstLeadingBit(i16(-32768)) == i16(14);
        const_assert firstTrailingBit(u16(0)) == u16(65535);
        const_assert reverseBits(i16(1)) == i16(-32768);
        const_assert extractBits(i16(-32768), 12u, 4u) == i16(-8);
        const_assert insertBits(u16(7), u16(0xabc), 12u, 4u) == u16(0xc007);
        const_assert dot(vec2<i16>(i16(-2), i16(3)), vec2<i16>(i16(4), i16(5))) == i16(7);
        """)
    _ = try IndependentShaderCompiler().parseAndValidate(source, capabilities: .referenceDefault.union(.shaderInt16))
}

@Test(arguments: ["extractBits(u16(1), 15u, 2u)", "insertBits(i16(0), i16(-1), 17u, 0u)",
                  "extractBits(i16(1), u16(0), 1u)", "dot(vec2<i16>(i16(1)), vec2<u16>(u16(1)))"])
func wgslInt16BuiltinsRejectInvalidRangesAndTypes(_ expression: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl("enable wgpu_int16; const result = \(expression);"),
                                                         capabilities: .referenceDefault.union(.shaderInt16))
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func wgslInt16BuiltinsPreserveBitsAndWrappingDotsOnMetal(_ roundTripSPIRV: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shaderInput: ShaderInput
    if roundTripSPIRV {
        guard case .spirv(let words) = try compiler.compile(wgslInt16BuiltinInput, to: .spirv) else {
            throw ShaderFailure.incompatibleArtifact
        }
        shaderInput = .spirv(words)
    } else {
        shaderInput = wgslInt16BuiltinInput
    }
    let shader = try compiler.compileCompute(shaderInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 48, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 224, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 224, usage: [.copyDestination, .mapRead]))
        let seeds: [UInt32] = [0, 1, 0x8000, 0xffff, 12, .max, .max, 0, 3, 1, 2, 3]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: seeds),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 224), .read(buffer: readback, byteCount: 224),
        ]).wait()
        let expected: [[UInt32]] = [
            [16, 15, 0, 0], [16, 0, 15, 0], [0, 1, 1, 16], [0, 0xffff8000, 1, .max],
            [0xffff, 0, 15, 15], [.max, 0, 14, .max], [.max, 0, 15, 0], [0, 0, 8, 15],
            [0, 0, 0xfffffff8, .max], [0xc000, 0xc001, 0xc000, 0xcfff],
            [0, 1, 0xffff8000, .max], [0, 1, 0xffff8000, .max],
            [0, 0xffff, 16, 0xffff], [0x7fff, 0xfffd, 0xffff8003, 16],
        ]
        #expect(snapshots == [expected.flatMap { $0 }])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
