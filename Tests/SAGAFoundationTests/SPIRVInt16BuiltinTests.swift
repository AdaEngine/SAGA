import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvInt16BuiltinInput = ShaderInput.wgsl("""
    enable wgpu_int16;
    struct Input { values: vec4<u16>, ranges: vec2u }
    @group(0) @binding(0) var<storage, read> input: Input;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 10>;
    @compute @workgroup_size(1) fn run() {
        let values = input.values;
        let signed = bitcast<vec4<i16>>(values);
        let range = input.ranges;
        output[0] = vec4u(countLeadingZeros(values));
        output[1] = vec4u(countTrailingZeros(values));
        output[2] = vec4u(countOneBits(signed));
        output[3] = vec4u(bitcast<vec4<u16>>(reverseBits(signed)));
        output[4] = vec4u(firstLeadingBit(values));
        output[5] = vec4u(bitcast<vec4<u16>>(firstLeadingBit(signed)));
        output[6] = vec4u(bitcast<vec4<u16>>(firstTrailingBit(signed)));
        output[7] = vec4u(bitcast<vec4<u16>>(extractBits(signed, range.x, range.y)));
        output[8] = vec4u(insertBits(values, vec4<u16>(u16(5)), range.x, range.y));
        output[9] = vec4u(u32(dot(values, vec4<u16>(u16(3)))), bitcast<u32>(i32(dot(signed, vec4<i16>(i16(-2))))),
                          u32(extractBits(values.x, 0u, range.y)), u32(insertBits(values.y, u16(5), range.y, 0u)));
    }
    """)

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvInt16BuiltinsRoundTripWithoutNarrowBitwiseInstructions(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(spirvInt16BuiltinInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    var capabilities: Set<UInt32> = []
    var widths: [UInt32: UInt32] = [:]
    var opcodes: Set<UInt32> = []
    var signedHalf: UInt32?
    var signExtendedLiteral = false
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        switch opcode {
        case 17: capabilities.insert(words[offset + 1])
        case 21:
            widths[words[offset + 1]] = words[offset + 2]
            if words[offset + 2] == 16, words[offset + 3] == 1 { signedHalf = words[offset + 1] }
        case 23: widths[words[offset + 1]] = widths[words[offset + 2]]
        case 43 where words[offset + 1] == signedHalf:
            if words[offset + 3] == 0xfffffffe { signExtendedLiteral = true }
        case 201...205:
            #expect(widths[words[offset + 1]] == 32)
        case 12 where (73...75).contains(words[offset + 4]):
            #expect(widths[words[offset + 1]] == 32)
        default: break
        }
        offset += count
    }
    #expect(capabilities.isSuperset(of: [22, 4433]))
    #expect(opcodes.isSuperset(of: [128, 132, 201, 202, 203, 204, 205]))
    #expect(!opcodes.contains(148))
    #expect(signExtendedLiteral)
    let artifact = try compiler.compileCompute(.spirv(words), to: format)
    #expect(artifact.requiresShaderInt16)
    #expect(artifact.storageBindings.map(\.minimumByteSize) == [16, 160])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.shaderInt16])
    }
}

#if canImport(Metal)
@Test func spirvInt16BuiltinsPreserveWordBoundariesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(spirvInt16BuiltinInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.shaderInt16) else { await device.close(); return }
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 160, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 160, usage: [.copyDestination, .mapRead]))
        let expected: [UInt32] = [
            16, 0, 0, 9, 16, 15, 0, 6, 0, 1, 16, 1, 0, 1, 0xffff, 0x0200,
            0xffff, 15, 15, 6, 0xffff, 14, 0xffff, 6, 0xffff, 15, 0, 6,
            0, 0xfff8, 0xffff, 0, 0x5000, 0x5000, 0x5fff, 0x5040, 0x80bd, 0xffffff82, 0, 0x8000,
        ]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0x80000000, 0x0040ffff, 12, .max]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 160), .read(buffer: readback, byteCount: 160),
        ])
        #expect(try await ticket.wait() == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
