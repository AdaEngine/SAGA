import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvInt64ArithmeticInput = ShaderInput.wgsl("""
    struct Operands {
        signedValues: vec2<i64>,
        signedDivisors: vec2<i64>,
        unsignedValues: vec2<u64>,
        unsignedDivisors: vec2<u64>,
        counts: vec4u,
    }
    @group(0) @binding(0) var<storage, read> input: Operands;
    @group(0) @binding(1) var<storage, read_write> output: array<vec2<u64>, 17>;

    @compute @workgroup_size(1) fn run() {
        let a = input.signedValues;
        let b = input.signedDivisors;
        let x = input.unsignedValues;
        let y = input.unsignedDivisors;
        let counts = input.counts;
        output[0] = bitcast<vec2<u64>>(a / b);
        output[1] = bitcast<vec2<u64>>(a % b);
        output[2] = bitcast<vec2<u64>>(a << counts.xy);
        output[3] = bitcast<vec2<u64>>(a >> counts.xy);
        output[4] = x << counts.xy;
        output[5] = x >> counts.xy;
        output[6] = x / y;
        output[7] = x % y;
        output[8] = bitcast<vec2<u64>>(-a);
        output[9] = bitcast<vec2<u64>>(a + b);
        output[10] = x + y;
        output[11] = bitcast<vec2<u64>>(a * b);
        output[12] = vec2<u64>(select(0lu, bitcast<u64>(-9223372036854775808li), a.x < a.y),
                               select(0lu, 18446744073709551615lu, x.y > y.y));
        output[13] = vec2<u64>(x.y << counts.z, bitcast<u64>(a.y >> counts.z));
        let divisor = i64(counts.w);
        output[14] = vec2<u64>(bitcast<u64>(a.y / divisor), bitcast<u64>(a.y % divisor));
        output[15] = vec2<u64>(u64(u32(x.y)), bitcast<u64>(i64(i32(a.y))));
        output[16] = vec2<u64>(u64(i64(counts.z)), u64(i32(a.y)));
    }
    """)

private func exportInt64Arithmetic(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvInt64ArithmeticInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvInt64ArithmeticDeclaresWideTypesAndGuardsRuntimeOperands() throws {
    let words = try exportInt64Arithmetic(IndependentShaderCompiler())
    var capabilities: Set<UInt32> = []
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
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 134 where count == 5, 137 where count == 5: divisors.append(words[offset + 4])
        case 169 where count == 6: guardedDivisors.insert(words[offset + 2]) // OpSelect
        case 194...196 where count == 5: shiftCounts.append(words[offset + 4])
        case 199 where count == 5: maskedCounts.insert(words[offset + 2]) // OpBitwiseAnd
        default: break
        }
        offset += count
    }
    #expect(capabilities.contains(11)) // Int64
    #expect(!capabilities.contains(12)) // Int64Atomics
    #expect(opcodes.isSuperset(of: [113, 114, 124, 134, 137, 172, 177, 194, 195, 196]))
    #expect(opcodes.isDisjoint(with: [135, 138, 139])) // Signed quotient uses unsigned magnitudes.
    #expect(shiftCounts.count == 6 && shiftCounts.allSatisfy(maskedCounts.contains))
    #expect(divisors.count == 6 && divisors.allSatisfy(guardedDivisors.contains))
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvInt64ArithmeticRoundTripsNativeShaderFormats(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportInt64Arithmetic(compiler)), to: format)
    #expect(shader.requiresShaderInt64)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [80, 272])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.shaderInt64))
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvInt64ArithmeticPreservesHighBitsAndRuntimeEdgeResultsOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shaderInput = try roundTrip ? ShaderInput.spirv(exportInt64Arithmetic(compiler)) : spirvInt64ArithmeticInput
    let shader = try compiler.compileCompute(shaderInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt64])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 80, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 272, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 272, usage: [.mapRead, .copyDestination]))
        func words(_ values: [UInt64]) -> [UInt32] {
            values.flatMap { [UInt32(truncatingIfNeeded: $0), UInt32($0 >> 32)] }
        }
        func bits(_ value: Int64) -> UInt64 { UInt64(bitPattern: value) }
        let high: UInt64 = 0x8000_0000_0000_0000
        let seeds = words([high, bits(-91), .max, 7, .max, high | 1, 0, 0x4000_0000_0000_0001]) + [64, 65, .max, 0]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: seeds),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 272), .read(buffer: readback, byteCount: 272),
        ]).wait()
        let expected: [UInt64] = [
            high, bits(-13), 0, 0, high, bits(-182), high, bits(-46),
            .max, 2, .max, 0x4000_0000_0000_0000, .max, 1, 0, 0x4000_0000_0000_0000,
            high, 91, high - 1, bits(-84), .max, 0xc000_0000_0000_0002, high, bits(-637),
            high, .max, high, .max, bits(-91), 0, 1, bits(-91), 0xffff_ffff, bits(-91),
        ]
        #expect(snapshots == [words(expected)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
