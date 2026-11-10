import Testing
import SAGACompiler
import SAGAShaderIR

@Test func hlslTaskEnforcesAmplificationWorkgroupLimit() throws {
    func source(_ size: String) -> String {
        """
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(\(size))
        fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
        """
    }
    let compiler = IndependentShaderCompiler()
    guard case .text(_, let hlsl) = try compiler.compileTaskStage(.wgsl(source("8, 4, 4")), entryPoint: "main", to: .hlsl) else {
        Issue.record("Expected HLSL"); return
    }
    #expect(hlsl.contains("[numthreads(8, 4, 4)]"))
    for size in ["129", "8, 8, 3"] {
        #expect(throws: ShaderFailure.unsupported("HLSL task workgroup size exceeds numthreads limits")) {
            try compiler.compileTaskStage(.wgsl(source(size)), entryPoint: "main", to: .hlsl)
        }
    }
}
