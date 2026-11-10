#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskAndMeshMultidimensionalInvocationSizesReadBack() async throws {
    let taskSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
        @task @payload(payload) @workgroup_size(2, 2, 2)
        fn task() -> @builtin(mesh_task_size) vec3u {
            atomicAdd(&counter, 1u);
            return vec3u(2u, 2u, 1u);
        }
        """)
    let meshSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output {
            @builtin(vertices) vertices: array<V, 3>,
            @builtin(primitives) primitives: array<P, 1>,
            @builtin(vertex_count) vertex_count: u32,
            @builtin(primitive_count) primitive_count: u32,
        }
        var<workgroup> output: Output;
        @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
        @mesh(output) @payload(payload) @workgroup_size(3, 2, 1)
        fn mesh(@builtin(local_invocation_index) lane: u32) {
            atomicAdd(&counter, 1u);
            if lane == 0u { output.vertex_count = 0u; output.primitive_count = 0u; }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(taskSource, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(meshSource, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(0u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: .renderAttachment))
        let taskCounter = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource, .copyDestination]))
        let meshCounter = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .write(buffer: taskCounter, offset: 0, words: [0]),
            .write(buffer: meshCounter, offset: 0, words: [0]),
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                      taskBuffers: [.init(binding: 0): .init(buffer: taskCounter)],
                      meshBuffers: [.init(binding: 0): .init(buffer: meshCounter)]),
            .copy(source: taskCounter, destination: readback, byteCount: 4),
            .copy(source: meshCounter, destination: readback, destinationOffset: 4, byteCount: 4),
            .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(result == [[8, 24]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
