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
func metalTaskAndMeshWorkgroupArraysAndAtomicsReadBack() async throws {
    let taskSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        var<workgroup> values: array<u32, 1024>;
        var<workgroup> sum: atomic<u32>;
        @task @payload(payload) @workgroup_size(4)
        fn task(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
            values[1020u + lane] = lane + 1u;
            workgroupBarrier();
            atomicAdd(&sum, values[1020u + lane]);
            workgroupBarrier();
            if lane == 0u { payload.value = atomicLoad(&sum); }
            return vec3u(1u);
        }
        """)
    let meshSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        var<workgroup> values: array<u32, 2048>;
        var<workgroup> sum: atomic<u32>;
        @mesh(output) @payload(payload) @workgroup_size(4)
        fn mesh(@builtin(local_invocation_index) lane: u32) {
            values[2044u + lane] = payload.value + lane;
            workgroupBarrier();
            atomicAdd(&sum, values[2044u + lane]);
            workgroupBarrier();
            if lane == 0u && atomicLoad(&sum) == 46u {
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
