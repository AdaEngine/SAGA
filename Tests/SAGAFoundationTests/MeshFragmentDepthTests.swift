#if canImport(Metal)
import Testing
import SAGATypes
import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGAMetal
import Metal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshFragmentDepthAndStencilReadback() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(1) fn mesh() {
          output.vertex_count = 3u; output.primitive_count = 1u;
          output.vertices[0].position = vec4f(-1.0, -1.0, 0.5, 1.0);
          output.vertices[1].position = vec4f(3.0, -1.0, 0.5, 1.0);
          output.vertices[2].position = vec4f(-1.0, 3.0, 0.5, 1.0);
          output.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("""
        struct Out { @builtin(frag_depth) depth: f32, @location(0) color: vec4u, }
        @fragment fn fs() -> Out { return Out(0.25, vec4u(1u, 2u, 3u, 255u)); }
        """), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        await #expect(throws: GPUError.self) {
            _ = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        }
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint,
            depthState: .init(compare: .less, writeEnabled: true))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float, usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: color, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), depth: .init(texture: depth, loadAction: .clear(0.4))),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff030201], [Float(0.25).bitPattern]])
        let replace = RenderStencilFaceState(pass: .replace)
        let equal = RenderStencilFaceState(compare: .equal)
        let writer = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint,
            depthState: .init(format: .depth24PlusStencil8, compare: .always, stencil: .init(front: replace, back: replace)))
        let reader = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint,
            depthState: .init(format: .depth24PlusStencil8, compare: .always, writeEnabled: false, stencil: .init(front: equal, back: equal)))
        let combined = try await device.createTexture(.init(width: 1, height: 1, format: .depth24PlusStencil8, usage: .renderAttachment))
        let stencilResult = try await device.submitCommands([
            .drawMesh(pipeline: writer, target: color, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                depth: .init(texture: combined, loadAction: .clear(1), stencil: .init(loadAction: .clear(0), reference: 7))),
            .drawMesh(pipeline: reader, target: color, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                depth: .init(texture: combined, loadAction: .readOnly, stencil: .init(loadAction: .readOnly, reference: 7))),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .drawMesh(pipeline: reader, target: color, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                depth: .init(texture: combined, loadAction: .readOnly, stencil: .init(loadAction: .readOnly, reference: 8))),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        ]).wait()
        #expect(stencilResult == [[0xff030201], [0]])
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
