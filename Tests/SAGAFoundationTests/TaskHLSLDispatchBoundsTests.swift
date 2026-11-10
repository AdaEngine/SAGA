import Testing
import SAGACompiler
import SAGAShaderIR

@Test func hlslTaskDispatchUsesOptionalDeviceLimits() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(4)
    fn main(@builtin(workgroup_id) id: vec3u) -> @builtin(mesh_task_size) vec3u { return id; }
    """
    let compiler = IndependentShaderCompiler()
    guard case .text(_, let unchecked) = try compiler.compileTaskStage(.wgsl(source), entryPoint: "main", to: .hlsl),
          case .text(_, let checked) = try compiler.compileTaskStage(
            .wgsl(source), entryPoint: "main", to: .hlsl,
            hlslTaskDispatchLimits: .init(maxMeshWorkgroupsPerDimension: 8_000_000, maxMeshWorkgroupsTotal: 1000)
          ) else { throw ShaderFailure.incompatibleArtifact }
    #expect(!unchecked.contains("(uint64_t)saga_size.x"))
    #expect(checked.contains("saga_size.x > 4194304u"))
    #expect(checked.contains("saga_size.y > 4194304u"))
    #expect(checked.contains("saga_size.z > 4194304u"))
    #expect(checked.contains("(uint64_t)saga_size.x * (uint64_t)saga_size.y > 1000u"))
    #expect(checked.contains("(uint64_t)saga_size.x * (uint64_t)saga_size.y * (uint64_t)saga_size.z > 1000u"))
    #expect(checked.contains("saga_size = uint3(0u, 0u, 0u)"))
    let guardRange = try #require(checked.range(of: "saga_size.x > 4194304u"))
    let saveRange = try #require(checked.range(of: "saga_dispatch_size = saga_size"))
    let dispatchRange = try #require(checked.range(of: "DispatchMesh("))
    #expect(guardRange.lowerBound < saveRange.lowerBound && saveRange.lowerBound < dispatchRange.lowerBound)
}
