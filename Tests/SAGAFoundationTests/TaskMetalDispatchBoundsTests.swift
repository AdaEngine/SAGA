import Testing
import SAGACompiler
import SAGAShaderIR

@Test func metalTaskDispatchChecksDimensionsAndOverflowBeforeDispatch() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(4)
    fn main(@builtin(workgroup_id) id: vec3u) -> @builtin(mesh_task_size) vec3u {
        return id;
    }
    """
    guard case .text(_, let text) = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: .msl) else {
        throw ShaderFailure.incompatibleArtifact
    }
    #expect(text.contains("#ifndef SAGA_MESH_GRID_LIMIT\n#define SAGA_MESH_GRID_LIMIT 1024u\n#endif"))
    #expect(text.contains("saga_size.x > SAGA_MESH_GRID_LIMIT"))
    #expect(text.contains("saga_size.y > SAGA_MESH_GRID_LIMIT"))
    #expect(text.contains("saga_size.z > SAGA_MESH_GRID_LIMIT"))
    #expect(text.contains("metal::mulhi(saga_size.x, saga_size.y) != 0u"))
    #expect(text.contains("metal::mulhi(saga_size.x * saga_size.y, saga_size.z) != 0u"))
    #expect(text.contains("saga_size.x * saga_size.y * saga_size.z > SAGA_MESH_GRID_LIMIT"))
    #expect(text.contains("saga_size = uint3(0u)"))
    #expect(text.contains("saga_grid.set_threadgroups_per_grid(saga_size)"))
}
