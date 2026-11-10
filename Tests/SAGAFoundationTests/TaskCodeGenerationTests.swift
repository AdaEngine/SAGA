import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskCodeGenerationUsesBodyAndPayload() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(4)
    fn main() -> @builtin(mesh_task_size) vec3u {
        payload.value = 7u;
        return vec3u(2u, 1u, 1u);
    }
    """
    for format in [ShaderOutputFormat.msl, .hlsl] {
        let artifact = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: format)
        guard case .text(_, let output) = artifact else { Issue.record("Expected text"); continue }
        #expect(output.contains("7u"))
        #expect(output.contains(format == .msl ? "set_threadgroups_per_grid" : "DispatchMesh"))
    }
}

@Test func taskCodeGenerationSupportsLocalInvocationIndex() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(4)
    fn main(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
        if lane == 0u { payload.value = 9u; }
        return vec3u(2u, 1u, 1u);
    }
    """
    for format in [ShaderOutputFormat.msl, .hlsl] {
        let artifact = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: format)
        guard case .text(_, let output) = artifact else { Issue.record("Expected text"); continue }
        #expect(output.contains("9u"))
        #expect(output.contains(format == .msl ? "thread_index_in_threadgroup" : "SV_GroupIndex"))
    }
}

@Test func taskCodeGenerationSupportsWorkgroupBarrier() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(4)
    fn main() -> @builtin(mesh_task_size) vec3u {
        workgroupBarrier();
        return vec3u(1u, 1u, 1u);
    }
    """
    for format in [ShaderOutputFormat.msl, .hlsl] {
        let artifact = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: format)
        guard case .text(_, let text) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(text.contains(format == .msl ? "threadgroup_barrier" : "GroupMemoryBarrierWithGroupSync"))
    }
}

@Test func taskArtifactReflectsValidatedPayload() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, other: i32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(4)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let result = try IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl)
    #expect(result.sourceEntryPoint == "main")
    #expect(result.entryPoint == "saga_main")
    #expect(result.workgroupSize == IRWorkgroupSize(x: 4))
    #expect(result.payload?.size == 8)
    #expect(result.payload?.members.map(\.offset) == [0, 4])
}

@Test func taskArtifactAcceptsMetalVectorPayloadLayout() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: vec4u, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(4)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let result = try IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl)
    #expect(result.payload?.size == 16)
    #expect(result.payload?.shape == .structure([.init(offset: 0, shape: .vector(scalar: .u32, count: 4))]))
}
