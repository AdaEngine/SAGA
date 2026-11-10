import Testing
import SAGACompiler
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation

@Test func meshMetadataRetainsOutputTypesWithOtherWorkgroupArrays() throws {
    let source = """
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        var<workgroup> task_shared: array<u32, 4>;
        @task @payload(payload) @workgroup_size(4)
        fn task(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
            if lane == 0u { payload.value = 7u; }
            workgroupBarrier();
            if lane == 1u { task_shared[0] = payload.value + task_shared[1]; }
            workgroupBarrier();
            if lane == 0u { payload.value = task_shared[0]; }
            return vec3u(1u);
        }
                struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        var<workgroup> mesh_shared: array<u32, 4>;
        @mesh(output) @payload(payload) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
            if lane == 1u { mesh_shared[0] = payload.value + mesh_shared[1]; }
            workgroupBarrier();
            if lane == 0u && mesh_shared[0] == 7u { output.vertex_count = 3u; output.primitive_count = 1u;
                output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
                output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
                output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
                output.primitives[0].indices = vec3u(0u, 1u, 2u); }
        }
        """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    let entry = try #require(module.entryPoints.first { $0.stage == .mesh })
    let info = try #require(entry.meshInfo)
    let output = try module.globalVariables.value(for: info.outputVariable)
    guard case .structure(let members, _) = try module.types.value(for: output.type).kind,
          case .array(let vertex, _, _) = try module.types.value(for: members[0].type).kind,
          case .array(let primitive, _, _) = try module.types.value(for: members[1].type).kind else { Issue.record("Invalid output"); return }
    #expect(info.vertexOutputType == vertex)
    #expect(info.primitiveOutputType == primitive)
    _ = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.meshShader]))
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileReflectedTaskStage(.wgsl(source), entryPoint: "task", to: .msl)
    _ = try compiler.compileReflectedMeshStage(.wgsl(source), entryPoint: "mesh", to: .msl)
}
