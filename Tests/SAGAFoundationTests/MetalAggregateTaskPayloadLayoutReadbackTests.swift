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
func metalAggregateTaskPayloadLayoutSurvivesMeshDispatch() async throws {
    // The mesh stage uses a deliberately different type: indexing its flat words
    // checks the actual offsets, rather than merely agreeing with task lowering.
    let taskSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Cell { @size(32) value: u32, }
        struct Payload {
            @align(32) head: u32,
            @size(32) marker: u32,
            cells: array<Cell, 2>,
            tail: u32,
        }
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(1)
        fn task() -> @builtin(mesh_task_size) vec3u {
            payload = Payload(11u, 23u, array<Cell, 2>(Cell(37u), Cell(53u)), 71u);
            return vec3u(1u);
        }
        """)
    let meshSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { words: array<u32, 32>, }
        var<task_payload> payload: Payload;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @payload(payload) @workgroup_size(1)
        fn mesh() {
            if payload.words[0] == 11u && payload.words[1] == 23u &&
                payload.words[9] == 37u && payload.words[17] == 53u && payload.words[25] == 71u {
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
