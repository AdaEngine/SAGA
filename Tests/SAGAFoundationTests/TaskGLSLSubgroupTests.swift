import Testing
import SAGACompiler

@Test func taskGLSLSubgroupsDeclareRequiredExtensions() throws {
    let source = """
    enable wgpu_mesh_shader;
    enable subgroups;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(32)
    fn main(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32) -> @builtin(mesh_task_size) vec3u {
        let total = subgroupAdd(lane);
        if lane == 0u { payload.value = total + size; }
        return vec3u(1u);
    }
    """
    let shader = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: .glsl)
    guard case .text(_, let text) = shader else { Issue.record("Expected GLSL source"); return }
    #expect(text.contains("#extension GL_KHR_shader_subgroup_basic : require"))
    #expect(text.contains("#extension GL_KHR_shader_subgroup_arithmetic : require"))
    #expect(text.contains("gl_SubgroupInvocationID"))
    #expect(text.contains("gl_SubgroupSize"))
    #expect(text.contains("subgroupAdd("))
}

@Test func taskGLSLSubgroupBarrierDeclaresBasicExtension() throws {
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
    let shader = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: .glsl)
    guard case .text(_, let text) = shader else { Issue.record("Expected GLSL source"); return }
    #expect(text.contains("#extension GL_KHR_shader_subgroup_basic : require"))
    #expect(text.contains("subgroupBarrier();"))
}
