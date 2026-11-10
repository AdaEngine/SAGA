#if canImport(Metal)
import Metal
import SAGACore
import SAGACompiler
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshResolvesAndWritesDisjointMips() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> out: O;
        @group(0) @binding(3) var image: texture_storage_2d<rgba8unorm, write>;
        @mesh(out) @workgroup_size(1) fn main() {
            textureStore(image, vec2i(0), vec4f(0.0, 1.0, 0.0, 1.0));
            out.vertex_count = 3u; out.primitive_count = 1u;
            out.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
            out.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
            out.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
            out.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "main", to: .msl)
    let fragment = try compiler.compileRenderStage(
        .wgsl("@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }"),
        entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm, sampleCount: 4)
        let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
            usage: .renderAttachment, sampleCount: 4))
        let image = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
            usage: [.renderAttachment, .storageBinding, .copySource], mipLevelCount: 2))
        let mip0 = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
        let readback = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                      meshTextures: [.init(binding: 3): mip0], resolve: .init(texture: image, mipLevel: 1)),
            .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: readback, byteCount: 264),
        ]).wait()
        #expect(result.count == 2)
        let storage = try #require(result.first)
        let resolved = try #require(result.last)
        #expect(storage == [0xff00ff00])
        try #require(resolved.count == 66)
        #expect([resolved[0], resolved[1], resolved[64], resolved[65]] == Array(repeating: 0xff0000ff, count: 4))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
