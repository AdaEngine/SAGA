import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskHLSLInitializesAndUsesWorkgroupMemory() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> values: array<u32, 8>;
    fn read_value() -> u32 { return values[0u]; }
    @task @payload(payload) @workgroup_size(4)
    fn main(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
        values[lane] = lane;
        workgroupBarrier();
        if lane == 0u { payload.value = read_value(); }
        return vec3u(lane + 1u, 1u, 1u);
    }
    """
    let result = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: .hlsl)
    guard case .text(_, let code) = result else { throw ShaderFailure.incompatibleArtifact }
    #expect(code.contains("groupshared uint saga_shared_"))
    #expect(code.contains("_init"))
    #expect(code.contains("saga_dispatch_size = saga_size;"))
    #expect(code.contains("DispatchMesh(saga_dispatch_size.x, saga_dispatch_size.y, saga_dispatch_size.z, saga_payload);"))
    let dispatch = try #require(code.range(of: "DispatchMesh("))
    let save = try #require(code.range(of: "saga_dispatch_size = saga_size;"))
    let afterSave = code[save.upperBound..<dispatch.lowerBound]
    #expect(afterSave.contains("GroupMemoryBarrierWithGroupSync();"))
}
