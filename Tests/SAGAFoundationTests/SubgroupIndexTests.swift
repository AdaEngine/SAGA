import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let subgroupIndexShader = """
    requires subgroup_id;
    enable subgroups;

    struct Subgroup {
        @builtin(subgroup_id) group: u32,
        @builtin(num_subgroups) count: u32,
    }
    @group(0) @binding(0) var<storage, read_write> records: array<vec4u, 260>;

    @compute @workgroup_size(130)
    fn run(info: Subgroup, @builtin(global_invocation_id) global: vec3u,
           @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32) {
        if info.count > 1u { workgroupBarrier(); }
        records[global.x] = vec4u(info.group, info.count, lane, size);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func subgroupIndexLowersWithExactWorkgroupMembership(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileCompute(.wgsl(subgroupIndexShader), to: format)
    #expect(artifact.requiresSubgroups && artifact.requiresSubgroupID)
    guard case .text(_, let code) = artifact.artifact else { Issue.record("Expected shader text"); return }
    switch format {
    case .wgsl:
        #expect(code.contains("requires subgroup_id;"))
        #expect(code.contains("@builtin(subgroup_id)") && code.contains("@builtin(num_subgroups)"))
        _ = try compiler.parseAndValidate(.wgsl(code), capabilities: [.subgroup])
    case .msl:
        #expect(code.contains("[[simdgroup_index_in_threadgroup]]") && code.contains("[[simdgroups_per_threadgroup]]"))
    case .hlsl:
        #expect(code.contains("SV_GroupIndex"))
        #expect(code.contains("InterlockedAdd(") && code.contains("WaveReadLaneFirst("))
        #expect(code.contains("GroupMemoryBarrierWithGroupSync()"))
    case .glsl:
        #expect(code.contains("gl_SubgroupID") && code.contains("gl_NumSubgroups"))
        #expect(code.contains("GL_KHR_shader_subgroup_basic"))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    "enable subgroups; @compute @workgroup_size(1) fn run(@builtin(subgroup_id) group: u32) {}",
    "requires subgroup_id; @compute @workgroup_size(1) fn run(@builtin(num_subgroups) count: u32) {}",
    "requires subgroup_id; enable subgroups; @compute @workgroup_size(1) fn run(@builtin(subgroup_id) group: i32) {}",
    "requires subgroup_id; enable subgroups; @fragment fn run(@builtin(num_subgroups) count: u32) {}",
])
func subgroupIndexRejectsMissingDirectivesAndInvalidInterfaces(source: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.subgroup])
    }
}

#if canImport(Metal)
@Test func subgroupIndicesReportMultipleGroupsAndPartialWavesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.subgroups, .subgroupID])
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(subgroupIndexShader), to: .msl))
    let records = try await device.createBuffer(.init(size: 4160, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4160, usage: [.copyDestination, .mapRead]))
    let snapshots = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): records], workgroups: .init(x: 2)),
        .copy(source: records, destination: readback, byteCount: 4160),
        .read(buffer: readback, byteCount: 4160),
    ]).wait()
    let words = try #require(snapshots.first)
    try #require(words.count == 1040)
    for workgroup in 0..<2 {
        let base = workgroup * 520
        let count = words[base + 1]
        let size = words[base + 3]
        try #require(count >= 2 && count <= 130 && size >= 4 && size <= 128)
        var lanes: [UInt32: Set<UInt32>] = [:]
        for index in stride(from: base, to: base + 520, by: 4) {
            let id = words[index]
            let lane = words[index + 2]
            #expect(words[index + 1] == count && words[index + 3] == size)
            #expect(id < count && lane < size && lanes[id, default: []].insert(lane).inserted)
        }
        #expect(Set(lanes.keys) == Set(0..<count))
        #expect(lanes.values.contains { $0.count < Int(size) })
    }
    await device.close()
}
#endif
