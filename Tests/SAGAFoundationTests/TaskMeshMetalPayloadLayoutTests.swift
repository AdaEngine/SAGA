import Testing
import SAGABackends
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation

@Test func metalTaskPayloadPreservesExplicitMemberLayout() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { @size(16) a: u32, @align(16) b: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1); }
    """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.meshShader]))
    guard case .text(_, let text) = try TaskCodeGenerator(format: .msl).generate(validated, entryPoint: "main") else {
        Issue.record("Expected text"); return
    }
    #expect(text.contains("uchar saga_padding_1[12];"))
    #expect(text.contains("uchar saga_padding_2[12];"))
}
