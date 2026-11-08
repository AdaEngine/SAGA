import Testing
import SAGABackend
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let linearIndexingShader = ShaderInput.wgsl("""
    requires linear_indexing;
    override width: u32 = 2u;
    struct Inputs {
        @builtin(workgroup_index) group_index: u32,
        @builtin(global_invocation_id) coordinate: vec3u,
        @builtin(num_workgroups) counts: vec3u,
    }
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u>;
    fn pack(value: vec3u) -> u32 { return value.x + 16u * value.y + 256u * value.z; }
    @compute @workgroup_size(width, 3, 2)
    fn run(input: Inputs, @builtin(global_invocation_index) index: u32,
           @builtin(workgroup_id) group: vec3u) {
        if group.x == 0u { workgroupBarrier(); }
        output[index] = vec4u(index, input.group_index, pack(input.coordinate),
                              pack(input.counts) + 1024u * pack(group));
    }
    """)

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsLinearIndicesWithSpecializedWorkgroupSize(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(linearIndexingShader, to: format, constants: ["width": 3])
    #expect(shader.workgroupSize.x == 3)
    #expect(shader.usesGlobalInvocationIndex && shader.usesWorkgroupIndex)
    guard case .text(_, let text) = shader.artifact else { Issue.record("Expected shader text"); return }
    switch format {
    case .wgsl:
        #expect(text.contains("requires linear_indexing;"))
        #expect(text.contains("@builtin(global_invocation_index)"))
        #expect(text.contains("@builtin(workgroup_index)"))
        _ = try compiler.parseAndValidate(.wgsl(text))
    case .msl:
        #expect(text.split(separator: "[[threadgroups_per_grid]]").count == 2)
        #expect(text.split(separator: "[[threadgroup_position_in_grid]]").count == 2)
        #expect(text.split(separator: "[[thread_position_in_grid]]").count == 2)
    case .hlsl:
        #expect(shader.hlslWorkgroupCountBinding == .init(group: 4, binding: 0))
        #expect(text.split(separator: "cbuffer SagaDispatch", omittingEmptySubsequences: false).count == 2)
        #expect(text.split(separator: "SV_GroupID").count == 2)
        #expect(text.split(separator: "SV_DispatchThreadID").count == 2)
    case .glsl:
        #expect(text.contains("gl_NumWorkGroups"))
        #expect(text.contains("gl_GlobalInvocationID"))
        #expect(text.contains("gl_WorkGroupID"))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    "@compute @workgroup_size(1) fn run(@builtin(workgroup_index) group: u32) {}",
    "requires linear_indexing; @compute @workgroup_size(1) fn run(@builtin(workgroup_index) group: vec3u) {}",
    "requires linear_indexing; @fragment fn run(@builtin(global_invocation_index) index: u32) {}",
    "requires linear_indexing; @compute @workgroup_size(2) fn run(@builtin(global_invocation_index) i: u32) { if i == 0u { workgroupBarrier(); } }",
    "requires linear_indexing; @compute @workgroup_size(2) fn run(@builtin(workgroup_index) i: u32) { if i == 0u { workgroupBarrier(); } }",
    "requires linear_indexing; @compute @workgroup_size(1) fn run(@builtin(workgroup_index) a: u32, @builtin(workgroup_index) b: u32) {}",
])
func rejectsInvalidLinearIndexingDeclarations(_ source: String) {
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

@Test(arguments: [false, true])
func linearIndexingValidatesTheRelevantDispatchLimit(global: Bool) throws {
    let builtin = global ? "global_invocation_index" : "workgroup_index"
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        requires linear_indexing;
        @compute @workgroup_size(2)
        fn run(@builtin(\(builtin)) index: u32) { _ = index; }
        """), to: .msl)
    #expect(shader.usesGlobalInvocationIndex == global)
    #expect(shader.usesWorkgroupIndex == !global)
    let pipeline = try ComputePipeline(id: .init(device: 1, serial: 1), shader: shader)
    #expect(pipeline.maximumLinearWorkgroupCount == (global ? UInt32.max / 2 : UInt32.max))
    let valid = DispatchWorkgroups(x: 65_535, y: global ? 32_768 : 65_535)
    let invalid = DispatchWorkgroups(x: 65_535, y: global ? 32_769 : 65_535, z: global ? 1 : 2)
    try BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: valid)
        .validate(device: 1, descriptors: [:])
    #expect(throws: GPUError.validation("Linear indexing dispatch exceeds u32")) {
        try BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: invalid)
            .validate(device: 1, descriptors: [:])
    }
}

#if canImport(Metal)
@Test func linearIndexingMatchesCPUForDirectAndIndirectMetalDispatches() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(linearIndexingShader, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let direct = DispatchWorkgroups(x: 2, y: 3, z: 2)
    let indirect = DispatchWorkgroups(x: 3, y: 2, z: 2)
    func expected(_ counts: DispatchWorkgroups) -> [UInt32] {
        var words: [UInt32] = []
        let packedCounts = counts.x + 16 * counts.y + 256 * counts.z
        for z in 0..<(2 * counts.z) {
            for y in 0..<(3 * counts.y) {
                for x in 0..<(2 * counts.x) {
                    let groupX = x / 2, groupY = y / 3, groupZ = z / 2
                    let groupIndex = groupX + counts.x * (groupY + counts.y * groupZ)
                    let packedGroup = groupX + 16 * groupY + 256 * groupZ
                    words += [UInt32(words.count / 4), groupIndex, x + 16 * y + 256 * z, packedCounts + 1024 * packedGroup]
                }
            }
        }
        return words
    }
    let first = expected(direct), second = expected(indirect)
    let byteCount = UInt64(first.count) * 4
    let output = try await device.createBuffer(.init(size: byteCount, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: byteCount, usage: [.mapRead, .copyDestination]))
    let arguments = try await device.createBuffer(.init(size: 24, usage: [.indirect, .copyDestination]))
    let slot = ComputeBindingSlot(binding: 0)
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [slot: output], workgroups: .init(x: 65_535, y: 65_535)),
        ])
    }
    let snapshot: [BufferCommand] = [
        .copy(source: output, destination: readback, byteCount: byteCount),
        .read(buffer: readback, byteCount: byteCount),
    ]
    let commands: [BufferCommand] = [
        .write(buffer: arguments, offset: 0, words: [indirect.x, indirect.y, indirect.z, 65_535, 65_535, 1]),
        .dispatchCompute(pipeline: pipeline, bindings: [slot: output], workgroups: direct),
    ] + snapshot + [
        .dispatchComputeIndirect(pipeline: pipeline, bindings: [slot: .init(buffer: output)], indirectBuffer: arguments),
    ] + snapshot + [
        .dispatchComputeIndirect(pipeline: pipeline, bindings: [slot: .init(buffer: output)],
                                 indirectBuffer: arguments, indirectOffset: 12),
    ] + snapshot
    let actual = try await device.submitCommands(commands).wait()
    #expect(actual == [first, second, second])
    await device.close()
}
#endif
