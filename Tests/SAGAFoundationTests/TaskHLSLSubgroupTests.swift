import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskHLSLSubgroupsAndLaneZeroDispatch() throws {
    let source = """
    requires subgroup_id;
    enable wgpu_mesh_shader, subgroups;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(32)
    fn main(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32,
            @builtin(subgroup_id) group: u32, @builtin(num_subgroups) count: u32) -> @builtin(mesh_task_size) vec3u {
        let sum = subgroupAdd(lane);
        if lane == 0u { payload.value = sum + size + group + count; }
        return vec3u(lane + 1u, 1u, 1u);
    }
    """
    let result = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: .hlsl)
    guard case .text(_, let code) = result else { throw ShaderFailure.incompatibleArtifact }
    #expect(code.contains("WaveGetLaneIndex()"))
    #expect(code.contains("WaveGetLaneCount()"))
    #expect(code.contains("WaveActiveSum("))
    #expect(code.contains("InterlockedAdd(saga_subgroup_count"))
    #expect(code.contains("WaveReadLaneFirst("))
    #expect(code.contains("groupshared uint3 saga_dispatch_size;"))
    #expect(code.contains("saga_dispatch_size = saga_size;"))
    #expect(code.contains("DispatchMesh(saga_dispatch_size.x, saga_dispatch_size.y, saga_dispatch_size.z, saga_payload);"))
}

@Test func taskHLSLRejectsSubgroupBarrierInHelper() {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    fn synchronize() { subgroupBarrier(); }
    @task @payload(payload) @workgroup_size(32)
    fn main() -> @builtin(mesh_task_size) vec3u {
        synchronize();
        return vec3u(1u);
    }
    """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: .hlsl)
    }
}
