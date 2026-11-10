import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskPayloadMetalPacksTightVectorsInNestedArrays() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Item { value: vec3u, tail: u32, }
    struct Payload { items: array<Item, 2>, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let artifact = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "main", to: .msl)
    guard case .text(_, let output) = artifact else { Issue.record("Expected Metal source"); return }
    #expect(output.contains("alignas(16) packed_uint3 saga_member_0;"))
}
