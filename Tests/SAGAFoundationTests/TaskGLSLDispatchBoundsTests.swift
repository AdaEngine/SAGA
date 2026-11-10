import Testing
import SAGACompiler
import SAGAShaderIR

@Test func glslTaskDispatchOptionallyChecksLimitsAndOverflow() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(4)
    fn main(@builtin(workgroup_id) id: vec3u) -> @builtin(mesh_task_size) vec3u { return id; }
    """
    let compiler = IndependentShaderCompiler()
    guard case .text(_, let unchecked) = try compiler.compileTaskStage(.wgsl(source), entryPoint: "main", to: .glsl),
          case .text(_, let checked) = try compiler.compileTaskStage(
            .wgsl(source), entryPoint: "main", to: .glsl,
            glslTaskDispatchLimits: .init(maxMeshWorkgroupsPerDimension: 1024, maxMeshWorkgroupsTotal: 4096)
          ) else { throw ShaderFailure.incompatibleArtifact }
    #expect(!unchecked.contains("umulExtended(saga_size.x"))
    #expect(checked.contains("umulExtended(saga_size.x, saga_size.y, saga_product_high, saga_product_low)"))
    #expect(checked.contains("umulExtended(saga_product_low, saga_size.z, saga_total_high, saga_total_low)"))
    #expect(checked.contains("saga_product_high != 0u || saga_total_high != 0u"))
    #expect(checked.contains("saga_size.x > 1024u || saga_size.y > 1024u || saga_size.z > 1024u"))
    #expect(checked.contains("saga_total_low > 4096u"))
    #expect(checked.contains("saga_size = uvec3(0u)"))
    let guardRange = try #require(checked.range(of: "umulExtended(saga_size.x"))
    let saveRange = try #require(checked.range(of: "saga_dispatch_size = saga_size"))
    let dispatchRange = try #require(checked.range(of: "EmitMeshTasksEXT("))
    #expect(guardRange.lowerBound < saveRange.lowerBound && saveRange.lowerBound < dispatchRange.lowerBound)
}

@Test func taskDispatchLimitsRejectMismatchedTarget() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) {
        try compiler.compileTaskStage(.wgsl(source), entryPoint: "main", to: .hlsl,
            glslTaskDispatchLimits: .init(maxMeshWorkgroupsPerDimension: 4, maxMeshWorkgroupsTotal: 4))
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileTaskStage(.wgsl(source), entryPoint: "main", to: .glsl,
            hlslTaskDispatchLimits: .init(maxMeshWorkgroupsPerDimension: 4, maxMeshWorkgroupsTotal: 4))
    }
}
