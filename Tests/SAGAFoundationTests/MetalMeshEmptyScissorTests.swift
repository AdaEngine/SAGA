#if canImport(Metal)
import Metal
import SAGACore
import SAGACompiler
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshEmptyScissorPreservesStorageWrites() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> out: O;
        @group(0) @binding(3) var image: texture_storage_2d<rgba8unorm, write>;
        @mesh(out) @workgroup_size(1) fn main() {
          textureStore(image, vec2i(0), vec4f(1.0, 0.0, 0.0, 1.0));
          out.vertex_count = 3u; out.primitive_count = 1u;
          out.vertices[0].position = vec4f(-1, -1, 0, 1);
          out.vertices[1].position = vec4f(3, -1, 0, 1);
          out.vertices[2].position = vec4f(-1, 3, 0, 1);
          out.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "main", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = []
        for scissor in [RenderScissor(x: 0, y: 0, width: 0, height: 1), RenderScissor(x: 0, y: 0, width: 1, height: 0)] {
            let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.storageBinding, .copySource]))
            commands += [
                .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                          meshTextures: [.init(binding: 3): image], raster: .init(scissor: scissor)),
                .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
                .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
            ]
        }
        let result = try await device.submitCommands(commands).wait()
        #expect(result == [[0xff0000ff], [0], [0xff0000ff], [0]])
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
