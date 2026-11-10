import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
#if canImport(Metal)
import Metal
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true), arguments: [false, true])
func metalTaskMeshFragmentStorageTextureShapes(volume: Bool) async throws {
    let textureType = volume ? "texture_storage_3d<rgba8uint, write>" : "texture_storage_2d_array<rgba8uint, write>"
    let coordinates = volume ? "vec3u(0)" : "vec2u(0), 0u"
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
          textureStore(taskImage, \(coordinates), vec4u(1u, 2u, 3u, 255u));
          payload.value = 1u; return vec3u(1u);
        }
        @mesh(output) @payload(payload) @workgroup_size(1) fn mesh() {
          textureStore(meshImage, \(coordinates), vec4u(4u, 5u, 6u, 255u));
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
          textureStore(fragmentImage, \(coordinates), vec4u(7u, 8u, 9u, 255u));
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
        let image = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Uint,
            usage: [.storageBinding, .copySource], mipLevelCount: volume ? 3 : 1,
            arrayLayerCount: volume ? 1 : 3, dimension: volume ? .d3 : .d2, depth: volume ? 4 : 1))
        var views: [ResourceID] = []
        for index in UInt32(0)..<3 {
            views.append(try await device.createTextureView(texture: image, volume
                ? .init(baseMipLevel: index, mipLevelCount: 1)
                : .init(dimension: .d2Array, baseArrayLayer: index, arrayLayerCount: 1)))
        }
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: .renderAttachment))
        let readback = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = [
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                      taskTextures: [.init(binding: 0): views[0]], meshTextures: [.init(binding: 1): views[1]],
                      fragmentTextures: [.init(binding: 2): views[2]])
        ]
        for index in UInt32(0)..<3 {
            commands += [.copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256,
                                               mipLevel: volume ? index : 0, arrayLayer: volume ? 0 : index),
                         .read(buffer: readback, byteCount: 4)]
        }
        let result = try await device.submitCommands(commands).wait()
        #expect(result == [[0xff030201], [0xff060504], [0xff090807]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
