import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
#if canImport(Metal)
import Metal
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskMeshFragmentStorageTexture1DReadback() async throws {
    let textureType = "texture_storage_1d<r32uint, write>"
    let coordinates = "1u"
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        @group(0) @binding(0) var taskImage: \(textureType);
        @group(0) @binding(1) var meshImage: \(textureType);
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: O;
        @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
          textureStore(taskImage, \(coordinates), vec4u(11u));
          payload.value = 1u; return vec3u(1u);
        }
        @mesh(output) @payload(payload) @workgroup_size(1) fn mesh() {
          textureStore(meshImage, \(coordinates), vec4u(22u));
          output.vertex_count = 3u; output.primitive_count = 1u;
          output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
          output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
          output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
          output.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        """)
    let fragmentSource = ShaderInput.wgsl("""
        @group(0) @binding(2) var fragmentImage: \(textureType);
        @fragment fn fs() -> @location(0) vec4u {
          textureStore(fragmentImage, \(coordinates), vec4u(33u));
          return vec4u(0u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(fragmentSource, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        var images: [ResourceID] = []
        for _ in 0..<3 {
            images.append(try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint,
                usage: [.storageBinding, .copySource], dimension: .d1)))
        }
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: .renderAttachment))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = [
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                      taskTextures: [.init(binding: 0): images[0]], meshTextures: [.init(binding: 1): images[1]],
                      fragmentTextures: [.init(binding: 2): images[2]])
        ]
        for image in images {
            commands += [.copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
                         .read(buffer: readback, byteCount: 8)]
        }
        let result = try await device.submitCommands(commands).wait()
        #expect(result.map { $0.last } == [11, 22, 33])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
