import Testing
import SAGACompiler
import SAGAShaderIR
#if canImport(Metal)
import Metal
import SAGACore
import SAGAMetal
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskMeshFragmentTextureAtomicsReadBack() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        @group(0) @binding(0) var taskCounts: texture_storage_2d<r32uint, atomic>;
        @group(0) @binding(1) var meshCounts: texture_storage_2d<r32uint, atomic>;
        @group(0) @binding(2) var fragmentCounts: texture_storage_2d<r32uint, atomic>;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(4)
        fn task(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
            textureAtomicAdd(taskCounts, vec2i(0, 0), 1u);
            if lane == 0u { payload.value = 1u; }
            return vec3u(1u);
        }
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: O;
        @mesh(output) @payload(payload) @workgroup_size(4)
        fn mesh(@builtin(local_invocation_index) lane: u32) {
            textureAtomicAdd(meshCounts, vec2i(0, 0), 1u);
            if lane == 0u {
                output.vertex_count = 3u; output.primitive_count = 1u;
                output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
                output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
                output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
                output.primitives[0].indices = vec3u(0u, 1u, 2u);
            }
        }
        @fragment fn fs() -> @location(0) vec4f {
            textureAtomicAdd(fragmentCounts, vec2i(0, 0), 1u);
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
        """)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.textureAtomic) else {
        await device.close()
        return
    }
    do {
        let compiler = IndependentShaderCompiler()
        let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
        let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let usage: TextureUsage = [.storageAtomic, .copySource]
        let taskCounts = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint, usage: usage))
        let meshCounts = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint, usage: usage))
        let fragmentCounts = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint, usage: usage))
        let target = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .copyTextureToBuffer(source: taskCounts, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .copyTextureToBuffer(source: meshCounts, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .copyTextureToBuffer(source: fragmentCounts, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), taskTextures: [.init(binding: 0): taskCounts], meshTextures: [.init(binding: 1): meshCounts], fragmentTextures: [.init(binding: 2): fragmentCounts]),
            .copyTextureToBuffer(source: taskCounts, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .copyTextureToBuffer(source: meshCounts, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .copyTextureToBuffer(source: fragmentCounts, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), taskTextures: [.init(binding: 0): taskCounts], meshTextures: [.init(binding: 1): meshCounts], fragmentTextures: [.init(binding: 2): fragmentCounts]),
            .copyTextureToBuffer(source: taskCounts, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .copyTextureToBuffer(source: meshCounts, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .copyTextureToBuffer(source: fragmentCounts, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(result == [[0, 0], [0, 0], [0, 0], [4, 0], [4, 0], [8, 0], [8, 0], [8, 0], [16, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
