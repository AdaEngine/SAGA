import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvSubgroupInput = ShaderInput.wgsl("""
    requires subgroup_id;
    enable subgroups;

    struct Subgroup {
        @builtin(subgroup_size) size: u32,
        @builtin(subgroup_invocation_id) lane: u32,
        @builtin(subgroup_id) group: u32,
        @builtin(num_subgroups) count: u32,
    }
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 1024>;

    fn paired(lane: u32) -> u32 { return subgroupShuffle(lane, lane ^ 1u); }

    @compute @workgroup_size(128) fn run(info: Subgroup, @builtin(local_invocation_index) index: u32) {
        let size = info.size;
        let lane = info.lane;
        let base = index * 8u;
        let factor = select(1u, 2u, lane < 3u);
        let down = select(lane, subgroupShuffleDown(lane, 1u), lane + 1u < size);
        let up = select(lane, subgroupShuffleUp(lane, 1u), lane != 0u);
        output[base] = vec4u(size, lane, info.group, info.count);
        output[base + 1u] = vec4u(subgroupBroadcast(lane + 10u, 0u), subgroupBroadcastFirst(lane + 20u),
                                  paired(lane), subgroupShuffleXor(lane, size - 1u));
        output[base + 2u] = subgroupBallot((lane & 1u) == 0u);
        output[base + 3u] = vec4u(subgroupAdd(lane), subgroupInclusiveAdd(lane), subgroupExclusiveAdd(lane), subgroupMul(factor));
        output[base + 4u] = vec4u(subgroupInclusiveMul(factor), subgroupExclusiveMul(factor),
                                  u32(subgroupAll(bool(size - lane))), u32(subgroupAny(lane == size - 1u)));
        output[base + 5u] = vec4u(down, up, subgroupMin(lane + 7u), subgroupMax(lane));
        output[base + 6u] = vec4u(subgroupAnd(lane | 0x80000000u), subgroupOr(lane), subgroupXor(lane), u32(subgroupElect()));
        output[base + 7u] = vec4u(quadBroadcast(lane, 0u), quadSwapX(lane), quadSwapY(lane), quadSwapDiagonal(lane));
    }
    """)

private let spirvFragmentSubgroupInput = ShaderInput.wgsl("""
    enable subgroups;
    @fragment fn fs(@builtin(subgroup_size) size: u32, @builtin(subgroup_invocation_id) lane: u32) -> @location(0) vec4u {
        return vec4u(size, lane, subgroupBroadcastFirst(lane), u32(subgroupAny(lane < size)));
    }
    """)

private func exportSubgroups(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvSubgroupInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvSubgroupCodeGeneratorDeclaresBuiltinsAndCollectiveModes() throws {
    let words = try exportSubgroups(IndependentShaderCompiler())
    var capabilities: Set<UInt32> = []
    var builtins: Set<UInt32> = []
    var opcodes: Set<UInt32> = []
    var modes: [UInt32: Set<UInt32>] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        switch opcode {
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 71 where count == 4 && words[offset + 2] == 11: builtins.insert(words[offset + 3])
        case 349 where count == 6, 351 where count == 6: modes[opcode, default: []].insert(words[offset + 4])
        default: break
        }
        offset += count
    }
    #expect(capabilities.isSuperset(of: [61, 62, 63, 64, 65, 66, 68]))
    #expect(builtins.isSuperset(of: [29, 36, 38, 40, 41]))
    #expect(opcodes.isSuperset(of: [334, 335, 337, 338, 339, 345, 346, 347, 348, 354, 357, 359, 360, 361, 365, 366]))
    #expect(modes[349] == [0, 1, 2] && modes[351] == [0, 1, 2])
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvSubgroupCodeGeneratorRoundTripsNativeShaderFormats(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportSubgroups(compiler)), to: format)
    #expect(shader.requiresSubgroups && shader.requiresSubgroupID)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [16384])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.subgroup))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvSubgroupCodeGeneratorRoundTripsFragmentSubgroupInputs(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(spirvFragmentSubgroupInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    var builtinTargets: Set<UInt32> = []
    var flatTargets: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        if (words[offset] & 0xffff) == 71 {
            if count == 4 && words[offset + 2] == 11 && [36, 41].contains(words[offset + 3]) {
                builtinTargets.insert(words[offset + 1])
            } else if count == 3 && words[offset + 2] == 14 {
                flatTargets.insert(words[offset + 1])
            }
        }
        offset += count
    }
    #expect(builtinTargets.count == 2 && builtinTargets.isSubset(of: flatTargets))
    let shader = try compiler.compileRenderStage(.spirv(words), entryPoint: "fs", to: format)
    #expect(shader.requiresSubgroups && shader.colorOutputComponentCounts == [0: 4])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.subgroup))
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvSubgroupCodeGeneratorPreservesLaneResultsOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shaderInput = try roundTrip ? ShaderInput.spirv(exportSubgroups(compiler)) : spirvSubgroupInput
    let shader = try compiler.compileCompute(shaderInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.subgroups, .subgroupID])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16384, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16384, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16384), .read(buffer: readback, byteCount: 16384),
        ]).wait()
        let words = try #require(snapshots.first)
        try #require(words.count == 4096)
        let size = words[0]
        let groupCount = words[3]
        try #require(size >= 4 && size <= 128 && size & (size - 1) == 0 && groupCount == 128 / size)
        let evenBallot = (0..<4).map { word -> UInt32 in
            let remaining = min(32, max(0, Int(size) - word * 32))
            let mask: UInt32 = remaining == 32 ? .max : (1 << remaining) - 1
            return mask & 0x55555555
        }
        var lanes: [UInt32: Set<UInt32>] = [:]
        for base in stride(from: 0, to: words.count, by: 32) {
            let lane = words[base + 1]
            let group = words[base + 2]
            try #require(lane < size && group < groupCount)
            #expect(words[base] == size && words[base + 3] == groupCount)
            #expect(lanes[group, default: []].insert(lane).inserted)
            let expected: [UInt32] = [10, 20, lane ^ 1, size - 1 - lane] + evenBallot + [
                size * (size - 1) / 2, lane * (lane + 1) / 2, lane * (lane &- 1) / 2, 8,
                1 << min(lane + 1, 3), 1 << min(lane, 3), 1, 1,
                min(lane + 1, size - 1), lane == 0 ? 0 : lane - 1, 7, size - 1,
                0x80000000, size - 1, 0, lane == 0 ? 1 : 0,
                lane & ~3, lane ^ 1, lane ^ 2, lane ^ 3,
            ]
            #expect(Array(words[(base + 4)..<(base + 32)]) == expected)
        }
        #expect(lanes.count == Int(groupCount) && lanes.values.allSatisfy { $0.count == Int(size) })
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
