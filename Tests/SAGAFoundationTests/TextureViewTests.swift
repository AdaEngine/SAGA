import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let textureViewShader = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f {
        return vec4f(p, 0.0, 1.0);
    }
    @fragment fn fs() -> @location(0) vec4f {
        return textureLoad(image, vec2u(0, 0), 0u);
    }
    """

@Test func resolvesMipViewRangesWithCheckedBounds() throws {
    let texture = TextureDescriptor(width: 4, height: 1, format: .rgba8Unorm, usage: .textureBinding, mipLevelCount: 3)
    #expect(try TextureViewDescriptor().mipLevels(in: texture) == 0..<3)
    #expect(try TextureViewDescriptor(baseMipLevel: 1).mipLevels(in: texture) == 1..<3)
    #expect(try TextureViewDescriptor(baseMipLevel: 1, mipLevelCount: 1).mipLevels(in: texture) == 1..<2)
    for descriptor in [
        TextureViewDescriptor(baseMipLevel: 3), .init(mipLevelCount: 0), .init(baseMipLevel: .max),
        .init(baseMipLevel: 2, mipLevelCount: 2), .init(baseMipLevel: 1, mipLevelCount: .max),
    ] {
        #expect(throws: GPUError.self) { try descriptor.mipLevels(in: texture) }
    }
}

#if canImport(Metal)
@Test func textureViewsSelectMipsAndPermitOnlyDisjointRenderBindingsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(textureViewShader)
        let render = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm, vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
        let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
            @group(0) @binding(0) var image: texture_2d<f32>;
            @group(0) @binding(1) var<storage, read_write> out: array<vec4f, 2>;
            @group(0) @binding(2) var state: sampler;
            @compute @workgroup_size(1) fn cs() {
                let p = textureLoad(image, vec2u(0), 0u);
                let q = textureLoad(image, vec2u(0), 1u);
                let invalid = textureLoad(image, vec2u(0), 2u);
                out[0] = vec4f(p.g, q.a + invalid.a, f32(textureDimensions(image).x), f32(textureNumLevels(image)));
                let left = textureSampleBaseClampToEdge(image, state, vec2f(-0.25, 0.25));
                let bottom = textureSampleBaseClampToEdge(image, state, vec2f(0.75, 1.25));
                let center = textureSampleBaseClampToEdge(image, state, vec2f(0.5));
                out[1] = vec4f(left.g, left.r, bottom.b, center.r);
            }
            """), to: .msl))
        let image = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination, .textureBinding, .renderAttachment], mipLevelCount: 3))
        let sampled = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
        let attachment = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
        let overlap = try await device.createTextureView(texture: image, .init(baseMipLevel: 1, mipLevelCount: 1))
        let state = try await device.createSampler(.init(addressModeU: .repeat, addressModeV: .repeat,
                                                         minFilter: .linear, magFilter: .linear, mipmapFilter: .linear))
        let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let out = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        let textureSlot = ComputeBindingSlot(binding: 0)
        func draw(to target: ResourceID) -> BufferCommand {
            .render(pipeline: render, target: target, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)], textures: [textureSlot: sampled])
        }
        await #expect(throws: GPUError.validation("Render attachment cannot be bound as a sampled texture")) {
            try await device.submitCommands([draw(to: overlap)])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([draw(to: sampled)])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.copyTextureToBuffer(source: attachment, destination: read, bytesPerRow: 256)])
        }
        let dispatch = BufferCommand.dispatchCompute(
            pipeline: compute, bindings: [.init(binding: 1): out], workgroups: .init(x: 1), textures: [textureSlot: sampled],
            samplers: [.init(binding: 2): state])
        let green: UInt32 = 0xff00ff00
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [green, 0xff0000ff]),
            .write(buffer: upload, offset: 256, words: [0xffff0000, 0xffffffff]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: image,
                                 region: .init(width: 2, height: 2, mipLevel: 1)),
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            draw(to: attachment),
            dispatch,
            .copy(source: out, destination: read, byteCount: 32),
            .read(buffer: read, byteCount: 32),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 16),
        ])
        let expected = [Float(1), 0, 2, 2, 1, 0, 1, 0.5].map(\.bitPattern)
        #expect(try await ticket.wait() == [expected, Array(repeating: green, count: 4)])
        let viewOnly = try await device.submitCommands([
            dispatch,
            .copy(source: out, destination: read, byteCount: 32),
            .read(buffer: read, byteCount: 32),
        ])
        try await device.destroyTexture(image)
        await #expect(throws: GPUError.validation("Texture view has a destroyed parent texture")) {
            try await device.submitCommands([dispatch])
        }
        try await device.destroyTextureView(sampled)
        try await device.destroyTextureView(attachment)
        #expect(try await viewOnly.wait() == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
