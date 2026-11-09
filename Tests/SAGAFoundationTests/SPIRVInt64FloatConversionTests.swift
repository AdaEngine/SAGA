import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvInt64FloatInput = ShaderInput.wgsl("""
    struct Operands {
        unsignedBounds: vec4f,
        signedBounds: vec4f,
        fractions: vec4f,
        signedValues: vec4<i64>,
        unsignedValues: vec4<u64>,
    }
    @group(0) @binding(0) var<storage, read> input: Operands;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4<u64>, 9>;

    @compute @workgroup_size(1) fn run() {
        let a = input.unsignedBounds;
        let b = input.signedBounds;
        let c = input.fractions;
        let s = input.signedValues;
        let u = input.unsignedValues;
        output[0] = vec4<u64>(a);
        output[1] = bitcast<vec4<u64>>(vec4<i64>(a));
        output[2] = bitcast<vec4<u64>>(vec4<i64>(b));
        output[3] = bitcast<vec4<u64>>(vec4<i64>(c));
        output[4] = vec4<u64>(c);
        output[5] = vec4<u64>(bitcast<vec4u>(vec4f(s)));
        output[6] = vec4<u64>(bitcast<vec4u>(vec4f(u)));
        output[7] = vec4<u64>(u64(a.y), bitcast<u64>(i64(b.w)), u64(c.y), bitcast<u64>(i64(c.y)));
        output[8] = vec4<u64>(u64(bitcast<u32>(f32(s.y))), u64(bitcast<u32>(f32(u.x))),
                               u64(bitcast<u32>(f32(s.w))), u64(bitcast<u32>(f32(u.z))));
    }
    """)

private func exportInt64FloatConversions(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvInt64FloatInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvInt64FloatConversionsRoundTripBothScalarAndVectorCasts(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportInt64FloatConversions(compiler)
    var capabilities: Set<UInt32> = []
    var opcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        if opcode == 17 && count == 2 { capabilities.insert(words[offset + 1]) }
        offset += count
    }
    #expect(opcodes.isSuperset(of: [109, 110, 111, 112])) // Float/integer conversion directions.
    #expect(capabilities.contains(11)) // Int64
    #expect(!capabilities.contains(10)) // Float64
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.requiresShaderInt64)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [128, 288])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.shaderInt64))
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvInt64FloatConversionsPreserveSaturationAndRoundingOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shaderInput = try roundTrip ? ShaderInput.spirv(exportInt64FloatConversions(compiler)) : spirvInt64FloatInput
    let shader = try compiler.compileCompute(shaderInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt64])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 128, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 288, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 288, usage: [.mapRead, .copyDestination]))
        func words(_ values: [UInt64]) -> [UInt32] {
            values.flatMap { [UInt32(truncatingIfNeeded: $0), UInt32($0 >> 32)] }
        }
        func floatBits(_ value: Float) -> UInt64 { UInt64(value.bitPattern) }
        let minimumSigned: UInt64 = 0x8000_0000_0000_0000
        let upperSigned: UInt64 = 0x7fff_ff80_0000_0000
        let upperUnsigned: UInt64 = 0xffff_ff00_0000_0000
        let floatSeeds: [UInt32] = [
            0x5f7fffff, 0x7f800000, 0xff800000, Float(-3.75).bitPattern,
            0x5effffff, 0xdf000000, 0x5f000000, 0xdf000001,
            Float(1.875).bitPattern, Float(-1.875).bitPattern, Float(0.5).bitPattern, Float(-0.5).bitPattern,
            0, 0, 0, 0,
        ]
        let integerSeeds: [UInt64] = [
            minimumSigned, upperSigned, 16_777_217, UInt64(bitPattern: -16_777_220),
            upperUnsigned, 0x8000_0100_0000_0000, 16_777_218, 16_777_220,
        ]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: floatSeeds + words(integerSeeds)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 288), .read(buffer: readback, byteCount: 288),
        ]).wait()
        let tieNeighbors: [Float] = [16_777_216, 16_777_218]
        let expected = tieNeighbors.map { rounded in
            words([
                upperUnsigned, upperUnsigned, 0, 0,
                upperSigned, upperSigned, minimumSigned, UInt64(bitPattern: -3),
                upperSigned, minimumSigned, upperSigned, minimumSigned,
                1, .max, 0, 0, 1, 0, 0, 0,
                0xdf000000, 0x5effffff, floatBits(rounded), floatBits(-16_777_220),
                0x5f7fffff, 0x5f000001, floatBits(16_777_218), floatBits(16_777_220),
                upperUnsigned, minimumSigned, 0, .max,
                0x5effffff, 0x5f7fffff, floatBits(-16_777_220), floatBits(16_777_218),
            ])
        }
        #expect(expected.contains { snapshots == [$0] })
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
