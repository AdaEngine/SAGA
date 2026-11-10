import SAGABackend
import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalNestedTaskPayloadSurvivesMeshDispatch() async throws {
    let taskSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Nested { vector: vec4f, values: array<u32, 2>, }
        struct Payload { nested: Nested, tail: u32, }
        var<task_payload> payload: Payload;
        fn initialize() {
            payload.nested.vector = vec4f(1.0, 2.0, 3.0, 4.0);
            payload.nested.values[0] = 11u;
            payload.nested.values[1] = 17u;
            payload.tail = 23u;
        }
        @task @payload(payload) @workgroup_size(2)
        fn task(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
            if lane == 0u { initialize(); }
            workgroupBarrier();
            if lane == 1u { payload.tail = payload.tail + payload.nested.values[0]; }
            workgroupBarrier();
            return vec3u(1u);
        }
        """)
    let meshSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Nested { vector: vec4f, values: array<u32, 2>, }
        struct Payload { nested: Nested, tail: u32, }
        var<task_payload> payload: Payload;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        var<workgroup> valid: u32;
        fn payload_is_valid() -> bool {
            return payload.nested.vector.x == 1.0 && payload.nested.vector.y == 2.0 &&
                payload.nested.vector.z == 3.0 && payload.nested.vector.w == 4.0 &&
                payload.nested.values[0] == 11u && payload.nested.values[1] == 17u && payload.tail == 34u;
        }
        @mesh(output) @payload(payload) @workgroup_size(2)
        fn mesh(@builtin(local_invocation_index) lane: u32) {
            if lane == 1u && payload_is_valid() { valid = 1u; }
            workgroupBarrier();
            if lane == 0u && valid == 1u {
                output.vertex_count = 3u; output.primitive_count = 1u;
                output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
                output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
                output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
                output.primitives[0].indices = vec3u(0u, 1u, 2u);
            }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(taskSource, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(meshSource, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(1u, 2u, 3u, 255u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1)),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff030201]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
