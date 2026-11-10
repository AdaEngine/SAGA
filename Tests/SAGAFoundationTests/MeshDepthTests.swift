import SAGATypes
import Testing

@Test func meshDepthAttachmentValidationAndResources() throws {
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm,
                                depthState: .init(compare: .less, writeEnabled: true))
    let color = ResourceID(device: 1, serial: 2)
    let depth = ResourceID(device: 1, serial: 3)
    let textures = [color: TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment]),
                    depth: TextureDescriptor(width: 4, height: 4, format: .depth32Float, usage: [.renderAttachment])]
    func draw(_ attachment: RenderDepthAttachment?) -> BufferCommand {
        .drawMesh(pipeline: pipeline, target: color, colorLoad: .load, workgroups: .init(x: 1), depth: attachment)
    }
    let valid = draw(.init(texture: depth))
    try valid.validate(device: 1, descriptors: [:], textures: textures)
    #expect(valid.resources.contains(depth))
    #expect(throws: GPUError.self) { try draw(nil).validate(device: 1, descriptors: [:], textures: textures) }
    #expect(throws: GPUError.self) { try draw(.init(texture: depth, loadAction: .readOnly)).validate(device: 1, descriptors: [:], textures: textures) }
    #expect(throws: GPUError.self) { try draw(.init(texture: depth, loadAction: .clear(2))).validate(device: 1, descriptors: [:], textures: textures) }
    #expect(throws: GPUError.self) { try draw(.init(texture: color)).validate(device: 1, descriptors: [:], textures: textures) }
}

#if canImport(Metal)
import Metal
import SAGAMetal
import SAGACore
import SAGACompiler
import SAGAShaderIR

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshDepthTestReadback() async throws {
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
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(1u, 2u, 3u, 255u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint, depthState: .init(compare: .less))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float, usage: [.renderAttachment]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: color, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), depth: .init(texture: depth, loadAction: .clear(0.25))),
            .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0]])
        let passing = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: color, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), depth: .init(texture: depth, loadAction: .clear(1))),
            .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(passing == [[0xff030201]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

@Test func meshStencilReadOnlyValidationAndPinning() throws {
    let color = ResourceID(device: 1, serial: 20)
    let depth = ResourceID(device: 1, serial: 21)
    let textures = [color: TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment),
                    depth: TextureDescriptor(width: 1, height: 1, format: .depth24PlusStencil8, usage: .renderAttachment)]
    let replace = RenderStencilFaceState(pass: .replace)
    let writer = MeshPipeline(id: .init(device: 1, serial: 22), colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth24PlusStencil8, stencil: .init(front: replace, back: replace)))
    let reader = MeshPipeline(id: .init(device: 1, serial: 23), colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth24PlusStencil8, writeEnabled: false, stencil: .init(front: .init(compare: .equal), back: .init(compare: .equal))))
    func draw(_ pipeline: MeshPipeline, _ attachment: RenderDepthAttachment) -> BufferCommand {
        .drawMesh(pipeline: pipeline, target: color, colorLoad: .load, workgroups: .init(x: 1), depth: attachment)
    }
    let attachment = RenderDepthAttachment(texture: depth, loadAction: .readOnly, stencil: .init(loadAction: .readOnly))
    let valid = draw(reader, attachment)
    try valid.validate(device: 1, descriptors: [:], textures: textures)
    #expect(valid.resources.contains(depth))
    #expect(throws: GPUError.self) { try draw(writer, attachment).validate(device: 1, descriptors: [:], textures: textures) }
    #expect(throws: GPUError.self) {
        try draw(reader, .init(texture: depth, loadAction: .readOnly, stencil: .init(loadAction: .readOnly, storeAction: .discard))).validate(device: 1, descriptors: [:], textures: textures)
    }
}
