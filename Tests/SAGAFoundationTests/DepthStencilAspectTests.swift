import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private func aspectShader(multisampled: Bool) -> String {
    let depthType = multisampled ? "texture_depth_multisampled_2d" : "texture_depth_2d"
    let stencilType = multisampled ? "texture_multisampled_2d<u32>" : "texture_2d<u32>"
    return """
        @group(0) @binding(0) var depthImage: \(depthType);
        @group(0) @binding(1) var stencilImage: \(stencilType);
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.25, 1); }
            if i == 1u { return vec4f(3, -1, 0.25, 1); }
            return vec4f(-1, 3, 0.25, 1);
        }
        @fragment fn seed() -> @location(0) vec4f { return vec4f(0); }
        @fragment fn useDepth() -> @location(0) vec4f {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), d > 0.24 && d < 0.26);
        }
        @fragment fn useStencil() -> @location(0) vec4f {
            let s = textureLoad(stencilImage, vec2u(0), 0u).r;
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), s == 7u);
        }
        """
}

private func aspectComputeShader(multisampled: Bool) -> String {
    let depthType = multisampled ? "texture_depth_multisampled_2d" : "texture_depth_2d"
    let stencilType = multisampled ? "texture_multisampled_2d<u32>" : "texture_2d<u32>"
    return """
        @group(0) @binding(0) var depthImage: \(depthType);
        @group(0) @binding(1) var stencilImage: \(stencilType);
        @group(0) @binding(2) var<storage, read_write> out: vec4u;
        @compute @workgroup_size(1) fn readAspects() {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            out = vec4u(select(0u, 1u, d > 0.24 && d < 0.26),
                        textureLoad(stencilImage, vec2u(0), 0u).r, textureDimensions(stencilImage));
        }
        """
}

@Test func validatesViewAspectsAgainstTextureFormat() throws {
    let color = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .textureBinding)
    let depth = TextureDescriptor(width: 1, height: 1, format: .depth32Float, usage: .textureBinding)
    #expect(throws: GPUError.validation("Texture view aspect does not match the texture format")) {
        _ = try TextureViewDescriptor(aspect: .depthOnly).mipLevels(in: color)
    }
    #expect(throws: GPUError.validation("Texture view aspect does not match the texture format")) {
        _ = try TextureViewDescriptor(aspect: .stencilOnly).mipLevels(in: depth)
    }
    #expect(try TextureViewDescriptor(aspect: .depthOnly).mipLevels(in: depth) == 0..<1)
}

#if canImport(Metal)
@Test(arguments: [UInt32(1), 4])
func samplesIndependentDepthStencilAspectsOnMetal(sampleCount: UInt32) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(aspectShader(multisampled: sampleCount > 1))
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let replace = RenderStencilFaceState(pass: .replace)
        let stencilWriter = RenderStencilState(front: replace, back: replace)
        func pipeline(_ entry: String, depthWrite: Bool, stencil: RenderStencilState? = nil) async throws -> RenderPipeline {
            try await device.createRenderPipeline(vertex: vertex,
                fragment: compiler.compileRenderStage(source, entryPoint: entry, to: .msl), colorFormat: .rgba8Unorm,
                depthState: .init(format: .depth24PlusStencil8, compare: .always, writeEnabled: depthWrite, stencil: stencil),
                sampleCount: sampleCount)
        }
        let seed = try await pipeline("seed", depthWrite: true, stencil: stencilWriter)
        let readDepth = try await pipeline("useDepth", depthWrite: false, stencil: stencilWriter)
        let readStencil = try await pipeline("useStencil", depthWrite: true)
        let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(aspectComputeShader(multisampled: sampleCount > 1)), to: .msl))
        let baseMip: UInt32 = sampleCount == 1 ? 1 : 0
        let combined = try await device.createTexture(.init(width: 2 << baseMip, height: 1 << baseMip, format: .depth24PlusStencil8,
            usage: [.renderAttachment, .textureBinding], mipLevelCount: baseMip + 1, sampleCount: sampleCount))
        let attachment = try await device.createTextureView(texture: combined, .init(baseMipLevel: baseMip))
        let depth = try await device.createTextureView(texture: combined, .init(baseMipLevel: baseMip, aspect: .depthOnly))
        let stencil = try await device.createTextureView(texture: combined, .init(baseMipLevel: baseMip, aspect: .stencilOnly))
        let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let target: ResourceID
        let resolve: RenderResolveAttachment?
        if sampleCount > 1 {
            target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                usage: .renderAttachment, sampleCount: sampleCount))
            resolve = .init(texture: color)
        } else {
            target = color
            resolve = nil
        }
        let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let copyColor = BufferCommand.copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256)
        let readColor = BufferCommand.read(buffer: read, byteCount: 8)
        await #expect(throws: GPUError.validation("Invalid sampled texture binding")) {
            _ = try await device.submitCommands([
                .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                                 textures: [.init(binding: 0): attachment, .init(binding: 1): stencil]),
            ])
        }
        await #expect(throws: GPUError.validation("Render attachment cannot be bound as a sampled texture")) {
            _ = try await device.submitCommands([
                .render(pipeline: readDepth, target: target, clear: .transparentBlack, vertexCount: 3,
                        textures: [.init(binding: 0): depth],
                        depth: .init(texture: attachment, loadAction: .load, stencil: .init(loadAction: .load)), resolve: resolve),
            ])
        }
        await #expect(throws: GPUError.validation("Invalid depth attachment format, extent or subresource")) {
            _ = try await device.submitCommands([
                .render(pipeline: seed, target: target, clear: .transparentBlack, vertexCount: 3,
                        depth: .init(texture: depth, stencil: .init()), resolve: resolve),
            ])
        }
        let ticket = try await device.submitCommands([
            .render(pipeline: seed, target: target, clear: .transparentBlack, vertexCount: 3,
                    depth: .init(texture: attachment, stencil: .init(reference: 173)), resolve: resolve),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
            .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
            .render(pipeline: readDepth, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 0): depth],
                    depth: .init(texture: attachment, loadAction: .readOnly,
                                 stencil: .init(loadAction: .load, reference: 7)), resolve: resolve),
            copyColor, readColor,
            .render(pipeline: readStencil, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 1): stencil],
                    depth: .init(texture: attachment, stencil: .init(loadAction: .readOnly)), resolve: resolve),
            copyColor, readColor,
        ])
        #expect(try await ticket.wait() == [[1, 173, 2, 1], [0xff00ff00, 0xff00ff00], [0xff00ff00, 0xff00ff00]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

@Test func validatesStencilCopyAspectAlignmentAndFullMipExtent() throws {
    let texture = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let textures = [texture: TextureDescriptor(width: 3, height: 2, format: .depth24PlusStencil8,
                                                usage: [.copySource, .copyDestination])]
    let buffers = [buffer: BufferDescriptor(size: 264, usage: [.copySource, .copyDestination])]
    func validate(_ command: BufferCommand) throws {
        try command.validate(device: 1, descriptors: buffers, textures: textures)
    }
    try validate(.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256,
                                      aspect: .stencilOnly, destinationOffset: 4))
    for aspect: TextureViewAspect in [.all, .depthOnly] {
        #expect(throws: GPUError.self) {
            try validate(.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256, aspect: aspect))
        }
    }
    #expect(throws: GPUError.self) {
        try validate(.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256,
                                          aspect: .stencilOnly, destinationOffset: 2))
    }
    #expect(throws: GPUError.self) {
        try validate(.copyBufferToTexture(source: buffer, layout: .init(bytesPerRow: 256), destination: texture,
                                          region: .init(width: 2, height: 2), aspect: .stencilOnly))
    }
    try validate(.copyBufferToTexture(source: buffer, layout: .init(offset: 4, bytesPerRow: 256), destination: texture,
                                      region: .init(width: 3, height: 2), aspect: .stencilOnly))
}

#if canImport(Metal)
@Test func copiesStencilBytesAndPreservesDepthOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(aspectShader(multisampled: false))
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let seed = try await device.createRenderPipeline(vertex: vertex,
            fragment: compiler.compileRenderStage(source, entryPoint: "seed", to: .msl), colorFormat: .rgba8Unorm,
            depthState: .init(format: .depth24PlusStencil8))
        let compute = try await device.createComputePipeline(
            compiler.compileCompute(.wgsl(aspectComputeShader(multisampled: false)), to: .msl))
        let combined = try await device.createTexture(.init(width: 6, height: 4, format: .depth24PlusStencil8,
            usage: [.renderAttachment, .textureBinding, .copySource, .copyDestination], mipLevelCount: 2))
        let attachment = try await device.createTextureView(texture: combined, .init(baseMipLevel: 1))
        let clone = try await device.createTexture(.init(width: 3, height: 2, format: .depth24PlusStencil8,
            usage: [.textureBinding, .copySource, .copyDestination]))
        let depth = try await device.createTextureView(texture: clone, .init(aspect: .depthOnly))
        let stencil = try await device.createTextureView(texture: clone, .init(aspect: .stencilOnly))
        let color = try await device.createTexture(.init(width: 3, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
        let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
        let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let ticket = try await device.submitCommands([
            .render(pipeline: seed, target: color, clear: .transparentBlack, vertexCount: 3,
                    depth: .init(texture: attachment, stencil: .init())),
            .write(buffer: upload, offset: 4, words: [0xaaf08001]),
            .write(buffer: upload, offset: 260, words: [0xbba50711]),
            .copyBufferToTexture(source: upload, layout: .init(offset: 4, bytesPerRow: 256), destination: combined,
                                 region: .init(width: 3, height: 2, mipLevel: 1), aspect: .stencilOnly),
            .copyTextureToTexture(source: combined, destination: clone, sourceRegion: .init(width: 3, height: 2, mipLevel: 1)),
            .copyTextureToBuffer(source: clone, destination: read, bytesPerRow: 256,
                                 aspect: .stencilOnly, destinationOffset: 4),
            .read(buffer: read, byteCount: 264),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
            .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ])
        var expected: [UInt32] = Array(repeating: 0, count: 66)
        expected[1] = 0x00f08001
        expected[65] = 0x00a50711
        #expect(try await ticket.wait() == [expected, [1, 1, 3, 2]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
