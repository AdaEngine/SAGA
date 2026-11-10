import Testing
import SAGACompiler
import SAGAShaderIR
#if canImport(Metal)
import Metal
import SAGACore
import SAGAMetal
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskMeshFragmentTextureInt64AtomicsReadBack() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        @group(0) @binding(0) var taskImage: texture_storage_2d<r64uint, atomic>;
        @group(0) @binding(1) var meshImage: texture_storage_2d<r64uint, atomic>;
        @group(0) @binding(2) var fragmentImage: texture_storage_2d<r64uint, atomic>;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(4)
        fn task(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
            textureAtomicMax(taskImage, vec2u(0), 0x123456789abcdef0lu);
            if lane == 0u { payload.value = 1u; }
            return vec3u(1u);
        }
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: O;
        @mesh(output) @payload(payload) @workgroup_size(4)
        fn mesh(@builtin(local_invocation_index) lane: u32) {
            textureAtomicMax(meshImage, vec2u(0), 0x87654321fedcba98lu);
            if lane == 0u {
                output.vertex_count = 3u; output.primitive_count = 1u;
                output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
                output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
                output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
                output.primitives[0].indices = vec3u(0u, 1u, 2u);
            }
        }
        @fragment fn fs() -> @location(0) vec4f {
            textureAtomicMax(fragmentImage, vec2u(0), 0xfedcba9876543210lu);
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
        """)
    let copySource = ShaderInput.wgsl("""
        @group(0) @binding(0) var input: texture_storage_2d<r64uint, read>;
        @group(0) @binding(1) var output: texture_storage_2d<r64uint, write>;
        @compute @workgroup_size(1) fn copy() { textureStore(output, vec2u(0), textureLoad(input, vec2u(0))); }
        """)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.textureInt64Atomic) else {
        await device.close()
        return
    }
    do {
        let compiler = IndependentShaderCompiler()
        let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
        let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let copy = try await device.createComputePipeline(compiler.compileCompute(copySource, entryPoint: "copy", to: .msl))
        let usage: TextureUsage = [.storageAtomic, .storageBinding]
        let taskImage = try await device.createTexture(.init(width: 1, height: 1, format: .r64Uint, usage: usage))
        let meshImage = try await device.createTexture(.init(width: 1, height: 1, format: .r64Uint, usage: usage))
        let fragmentImage = try await device.createTexture(.init(width: 1, height: 1, format: .r64Uint, usage: usage))
        let output = try await device.createTexture(.init(width: 1, height: 1, format: .r64Uint, usage: [.storageBinding, .copySource]))
        let target = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), taskTextures: [.init(binding: 0): taskImage], meshTextures: [.init(binding: 1): meshImage], fragmentTextures: [.init(binding: 2): fragmentImage]),
            .dispatchCompute(pipeline: copy, bindings: [:], workgroups: .init(x: 1), textures: [.init(binding: 0): taskImage, .init(binding: 1): output]),
            .copyTextureToBuffer(source: output, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
            .dispatchCompute(pipeline: copy, bindings: [:], workgroups: .init(x: 1), textures: [.init(binding: 0): meshImage, .init(binding: 1): output]),
            .copyTextureToBuffer(source: output, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
            .dispatchCompute(pipeline: copy, bindings: [:], workgroups: .init(x: 1), textures: [.init(binding: 0): fragmentImage, .init(binding: 1): output]),
            .copyTextureToBuffer(source: output, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(result == [[0x9abc_def0, 0x1234_5678], [0xfedc_ba98, 0x8765_4321], [0x7654_3210, 0xfedc_ba98]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
