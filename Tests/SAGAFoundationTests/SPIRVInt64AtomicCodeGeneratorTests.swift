import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvWideAtomicExtremaInput = ShaderInput.wgsl("""
    enable wgpu_binding_array;
    struct Values {
        minimum: atomic<u64>,
        maximum: atomic<u64>,
        tail: array<atomic<u64>>,
    }
    @group(0) @binding(0) var<storage, read_write> slots: binding_array<Values>;

    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let index = i32(lane) - 1i;
        atomicMin(&slots[index].minimum, 0x8000000000000000lu);
        atomicMax(&slots[index].maximum, 0xfedcba9876543210lu);
        atomicMax(&slots[index].tail[index], 0x9000000000000001lu);
        atomicMin(&slots[index].tail[i32(lane) - 4i], 0lu);
        atomicMax(&slots[index].tail[lane + 1u], 0xfffffffffffffffflu);
    }
    """)

private let wideAtomicLayout: [IRResourceBinding: UInt32] = [.init(group: 0, binding: 0): 2]

private enum WideAtomicScalar: String, CaseIterable, Sendable {
    case i64, u64
}

private func spirvWideAtomicFullInput(_ scalar: WideAtomicScalar) -> ShaderInput {
    let suffix = scalar == .i64 ? "li" : "lu"
    let initial = scalar == .i64 ? "-4li" : "0xfffffffffffffffclu"
    return .wgsl("""
        struct Counters { counter: atomic<\(scalar.rawValue)> }
        @group(0) @binding(0) var<storage, read_write> values: Counters;
        @group(0) @binding(1) var<storage, read_write> output: array<u64, 14>;

        @compute @workgroup_size(1) fn run() {
            atomicStore(&values.counter, \(initial));
            output[0] = bitcast<u64>(atomicLoad(&values.counter));
            output[1] = bitcast<u64>(atomicAdd(&values.counter, 7\(suffix)));
            output[2] = bitcast<u64>(atomicSub(&values.counter, 5\(suffix)));
            output[3] = bitcast<u64>(atomicMin(&values.counter, 2\(suffix)));
            output[4] = bitcast<u64>(atomicMax(&values.counter, 5\(suffix)));
            output[5] = bitcast<u64>(atomicAnd(&values.counter, 6\(suffix)));
            output[6] = bitcast<u64>(atomicOr(&values.counter, 3\(suffix)));
            output[7] = bitcast<u64>(atomicXor(&values.counter, 2\(suffix)));
            output[8] = bitcast<u64>(atomicExchange(&values.counter, 11\(suffix)));
            let match = atomicCompareExchangeWeak(&values.counter, 11\(suffix), 17\(suffix));
            output[9] = bitcast<u64>(match.old_value);
            output[10] = select(0lu, 1lu, match.exchanged);
            output[11] = bitcast<u64>(atomicLoad(&values.counter));
            let miss = atomicCompareExchangeWeak(&values.counter, 99\(suffix), 42\(suffix));
            output[12] = bitcast<u64>(miss.old_value);
            output[13] = select(0lu, 1lu, miss.exchanged);
        }
        """)
}

private func exportWideAtomic(
    _ input: ShaderInput, with compiler: IndependentShaderCompiler, layout: [IRResourceBinding: UInt32] = [:]
) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv, bindingArrayLengths: layout) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func checkWideAtomicInstructions(_ words: [UInt32], extremaOnly: Bool) throws -> Set<UInt32> {
    var capabilities: Set<UInt32> = []
    var opcodes: Set<UInt32> = []
    var atomicBlocks: [UInt32] = []
    var guardBlocks: Set<UInt32> = []
    var block: UInt32 = 0
    var hasRuntimeLength = false
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        switch opcode {
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 68: hasRuntimeLength = true // OpArrayLength
        case 227...242:
            opcodes.insert(opcode)
            atomicBlocks.append(block)
        case 248 where count == 2: block = words[offset + 1]
        case 250 where count >= 4: guardBlocks.insert(words[offset + 2])
        default: break
        }
        offset += count
    }
    #expect(capabilities.isSuperset(of: [11, 12])) // Int64 and Int64Atomics.
    if extremaOnly {
        #expect(hasRuntimeLength)
        #expect(atomicBlocks.count == 5 && atomicBlocks.allSatisfy(guardBlocks.contains))
    }
    return opcodes
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvInt64AtomicCodeGeneratorRoundTripsGuardedUnsignedExtrema(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportWideAtomic(spirvWideAtomicExtremaInput, with: compiler, layout: wideAtomicLayout)
    #expect(try checkWideAtomicInstructions(words, extremaOnly: true) == [237, 239])
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.requiresShaderInt64AtomicMinMax && !shader.requiresShaderInt64AtomicAllOps)
    let binding = try #require(shader.storageBindings.first)
    #expect(binding.bindingArray?.count == 2 && binding.minimumByteSize == 24)
    #expect(binding.runtimeArrayOffset == 16 && binding.runtimeArrayStride == 8)
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .allKnown)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .hlsl, .glsl], WideAtomicScalar.allCases)
private func spirvInt64AtomicCodeGeneratorRoundTripsOldValuesAndCompareExchange(
    _ format: ShaderOutputFormat, _ scalar: WideAtomicScalar
) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportWideAtomic(spirvWideAtomicFullInput(scalar), with: compiler)
    let extrema: Set<UInt32> = scalar == .i64 ? [236, 238] : [237, 239]
    #expect(try checkWideAtomicInstructions(words, extremaOnly: false) == extrema.union([227, 228, 229, 230, 234, 235, 240, 241, 242]))
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.requiresShaderInt64AtomicAllOps && !shader.requiresShaderInt64AtomicMinMax)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [8, 112])
    guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    switch format {
    case .wgsl:
        #expect(text.contains("atomic<\(scalar.rawValue)>") && text.contains("atomicCompareExchangeWeak("))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union([.shaderInt64, .shaderInt64AtomicAllOps]))
    case .hlsl: #expect(shader.hlslProfile == "cs_6_6" && text.contains("InterlockedCompareExchange64("))
    case .glsl: #expect(text.contains("atomicCompSwap("))
    default: Issue.record("Unexpected full-tier text target")
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvInt64AtomicCodeGeneratorPreservesHighBitsAndRuntimeBoundsOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shader: ComputeShaderArtifact
    if roundTrip {
        let words = try exportWideAtomic(spirvWideAtomicExtremaInput, with: compiler, layout: wideAtomicLayout)
        shader = try compiler.compileCompute(.spirv(words), to: .msl)
    } else {
        shader = try compiler.compileCompute(spirvWideAtomicExtremaInput, to: .msl, bindingArrayLengths: wideAtomicLayout)
    }
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.shaderInt64AtomicMinMax) else {
        await device.close()
        return
    }
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let storage = try await device.createBuffer(.init(size: 512, usage: [.storage, .copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
        func words(_ values: [UInt64]) -> [UInt32] {
            values.flatMap { [UInt32(truncatingIfNeeded: $0), UInt32($0 >> 32)] }
        }
        let belowSign: UInt64 = 0x7fff_ffff_0000_0001
        let sentinels: [UInt64] = [0x7abc_1234_1234_1234, 0x9abc_4321_4321_4321]
        let snapshots = try await device.submitCommands([
            .write(buffer: storage, offset: 0, words: words([.max, belowSign, 3, 7] + sentinels)),
            .write(buffer: storage, offset: 256, words: words([belowSign, .max, 8, 2] + sentinels)),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0, arrayElement: 0): .init(buffer: storage, size: 32),
                .init(binding: 0, arrayElement: 1): .init(buffer: storage, offset: 256, size: 32),
            ], workgroups: .init(x: 1)),
            .copy(source: storage, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
            .copy(source: storage, sourceOffset: 256, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
        ]).wait()
        #expect(snapshots == [
            words([0x8000_0000_0000_0000, 0xfedc_ba98_7654_3210, 0x9000_0000_0000_0001, 7] + sentinels),
            words([belowSign, .max, 8, 0x9000_0000_0000_0001] + sentinels),
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
