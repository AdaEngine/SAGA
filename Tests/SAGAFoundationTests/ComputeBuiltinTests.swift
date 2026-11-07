import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let invocationShader = """
    struct Invocation {
        @builtin(local_invocation_id) local: vec3u,
        @builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3u,
    }
    @group(0) @binding(2) var<storage, read_write> output: array<u32>;
    fn pack(v: vec3u) -> u32 { return v.x + v.y * 16u + v.z * 256u; }
    fn packLocal(v: Invocation) -> u32 { return pack(v.local); }
    @compute @workgroup_size(2, 3, 2)
    fn run(@builtin(global_invocation_id) global: vec3u, v: Invocation,
           @builtin(num_workgroups) count: vec3u) {
        let width = count.x * 2u;
        let height = count.y * 3u;
        let i = 5u * (global.x + width * (global.y + height * global.z));
        output[i] = pack(global);
        output[i + 1u] = packLocal(v);
        output[i + 2u] = v.lane;
        output[i + 3u] = pack(v.group);
        output[i + 4u] = pack(count);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func lowersComputeBuiltinsInDirectAndStructuredInputs(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(invocationShader), to: format)
    #expect(shader.workgroupSize == IRWorkgroupSize(x: 2, y: 3, z: 2))
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected text shader"); return }
    let expected: [String]
    switch format {
    case .msl:
        expected = ["thread_position_in_grid", "thread_position_in_threadgroup", "thread_index_in_threadgroup",
                    "threadgroup_position_in_grid", "threadgroups_per_grid"]
    case .hlsl:
        expected = ["SV_DispatchThreadID", "SV_GroupThreadID", "SV_GroupIndex", "SV_GroupID", "register(b0, space4)"]
        #expect(shader.hlslWorkgroupCountBinding == IRResourceBinding(group: 4, binding: 0))
    case .glsl:
        expected = ["gl_GlobalInvocationID", "gl_LocalInvocationID", "gl_LocalInvocationIndex", "gl_WorkGroupID",
                    "gl_NumWorkGroups"]
        _ = try compiler.parseAndValidate(.glsl(source, stage: .compute, defines: [:]))
    case .wgsl:
        expected = ["@builtin(global_invocation_id)", "@builtin(local_invocation_id)", "@builtin(local_invocation_index)",
                    "@builtin(workgroup_id)", "@builtin(num_workgroups)"]
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target"); return
    }
    if format != .hlsl { #expect(shader.hlslWorkgroupCountBinding == nil) }
    for spelling in expected { #expect(source.contains(spelling)) }
}

@Test func hlslSkipsWorkgroupCountBufferWhenBuiltinIsAbsent() throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        @compute @workgroup_size(1) fn run(@builtin(local_invocation_index) lane: u32) {}
        """), to: .hlsl)
    #expect(shader.hlslWorkgroupCountBinding == nil)
}

#if canImport(Metal)
@Test func computesBuiltinValuesAcrossTwoDispatchSizesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(invocationShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        for dispatch in [DispatchWorkgroups(x: 3, y: 2, z: 2), .init(x: 1, y: 2, z: 1)] {
            let expected = expectedInvocations(dispatch)
            let size = UInt64(expected.count * 4)
            let buffer = try await device.createBuffer(.init(size: size, usage: [.storage, .copySource]))
            let readback = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
            let ticket = try await device.submitCommands([
                .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): buffer], workgroups: dispatch),
                .copy(source: buffer, destination: readback, byteCount: size),
                .read(buffer: readback, byteCount: size),
            ])
            #expect(try await ticket.wait() == [expected])
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

private func expectedInvocations(_ dispatch: DispatchWorkgroups) -> [UInt32] {
    func pack(_ x: UInt32, _ y: UInt32, _ z: UInt32) -> UInt32 { x + y * 16 + z * 256 }
    var words: [UInt32] = []
    for z in 0..<(dispatch.z * 2) {
        for y in 0..<(dispatch.y * 3) {
            for x in 0..<(dispatch.x * 2) {
                let (lx, ly, lz) = (x % 2, y % 3, z % 2)
                words += [
                    pack(x, y, z), pack(lx, ly, lz), lx + 2 * ly + 6 * lz,
                    pack(x / 2, y / 3, z / 2), pack(dispatch.x, dispatch.y, dispatch.z),
                ]
            }
        }
    }
    return words
}
#endif
