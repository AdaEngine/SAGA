import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskGLSLInitializesReachableWorkgroupGlobals() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> scratch: array<u32, 7>;
    var<workgroup> counter: atomic<u32>;
    var<workgroup> unused: u32;
    fn fill() { scratch[0] = atomicAdd(&counter, 1u); }
    @task @payload(payload) @workgroup_size(4) fn task() -> @builtin(mesh_task_size) vec3u {
        fill();
        workgroupBarrier();
        payload.value = scratch[0];
        return vec3u(1u);
    }
    """
    let artifact = try IndependentShaderCompiler().compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl)
    guard case .text(_, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(code.contains("shared uint saga_shared_"))
    #expect(code.contains("_init < 7u"))
    #expect(code.contains("_init += 4u"))
    #expect(code.contains("atomicAdd("))
    #expect(code.contains("barrier();"))
    func occurrences(_ text: String) -> [String.Index] {
        code.indices.filter { code[$0...].hasPrefix(text) }
    }
    #expect(occurrences("shared uint saga_shared_").count == 2)
    let initialization = try #require(occurrences("_init += 4u").first)
    let barrier = try #require(occurrences("barrier();").first { $0 > initialization })
    let call = try #require(occurrences("saga_task_body(").first { $0 > barrier })
    #expect(initialization < barrier && barrier < call)
}
