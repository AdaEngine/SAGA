import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

@Test func validatesVolumeColorAttachmentDescriptors() throws {
    let limits = try DeviceCapabilities(maxBufferSize: 65536)
    try TextureDescriptor(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: 2, dimension: .d3, depth: 6).validate(against: limits)
    #expect(throws: GPUError.self) {
        try TextureDescriptor(width: 4, height: 2, format: .depth32Float,
            usage: .renderAttachment, dimension: .d3, depth: 6).validate(against: limits)
    }
    #expect(throws: GPUError.self) {
        try TextureDescriptor(width: 4, height: 2, format: .rgba8Unorm,
            usage: .renderAttachment, sampleCount: 4, dimension: .d3, depth: 6).validate(against: limits)
    }
}

private let volumeAttachmentShader = """
    @group(0) @binding(0) var input: texture_3d<f32>;
    struct FragmentOutput {
        @location(0) red: vec4f,
        @location(2) green: vec4f,
    }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> FragmentOutput {
        let red = textureLoad(input, vec3i(0), 0i).r;
        return FragmentOutput(vec4f(red, 0, 0, 1), vec4f(0, 1, 0, 1));
    }
    """

#if canImport(Metal)
@Test func rendersAndLoadsDisjointVolumeSlicesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(volumeAttachmentShader)
        let pipeline = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm, colorState: .init(writeMask: .red),
            additionalColorTargets: [2: .init(format: .rgba8Unorm, colorState: .init(writeMask: .green))])
        let texture = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: [.renderAttachment, .textureBinding, .copySource, .copyDestination], mipLevelCount: 2,
            dimension: .d3, depth: 6))
        let input = try await device.createTextureView(texture: texture, .init(mipLevelCount: 1))
        let output = try await device.createTextureView(texture: texture, .init(baseMipLevel: 1))
        let multiMip = try await device.createTextureView(texture: texture, .init())
        let flat = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
        let upload = try await device.createBuffer(.init(size: 12, usage: [.copySource, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 520, usage: [.copyDestination, .mapRead]))
        let firstColors: [UInt32: RenderColorAttachment] = [
            2: .init(texture: output, loadAction: .clear(.init(red: 0.25, green: 0, blue: 0, alpha: 1)), depthSlice: 1),
        ]
        func draw(target: ResourceID, slice: UInt32?, sampled: ResourceID, colors: [UInt32: RenderColorAttachment]) -> BufferCommand {
            .render(pipeline: pipeline, target: target, colorLoad: .load, vertexCount: 3,
                    textures: [.init(binding: 0): sampled], targetDepthSlice: slice, additionalColors: colors)
        }
        for invalid in [draw(target: output, slice: nil, sampled: input, colors: firstColors),
                        draw(target: output, slice: 3, sampled: input, colors: firstColors),
                        draw(target: flat, slice: 0, sampled: input, colors: firstColors),
                        draw(target: multiMip, slice: 0, sampled: input, colors: firstColors)] {
            await #expect(throws: GPUError.validation("Invalid render pipeline, color attachment or clear value")) {
                try await device.submitCommands([invalid])
            }
        }
        await #expect(throws: GPUError.validation("Aliased render attachments")) {
            try await device.submitCommands([draw(target: output, slice: 1, sampled: input, colors: firstColors)])
        }
        await #expect(throws: GPUError.validation("Render attachment cannot be bound as a sampled texture")) {
            try await device.submitCommands([draw(target: output, slice: 0, sampled: output, colors: firstColors)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ffff, 0xff00ffff]),
            .write(buffer: indices, offset: 0, words: [0, 1, 2]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: texture, region: .init(width: 1, height: 1)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 4), destination: texture,
                                 region: .init(width: 2, height: 1, mipLevel: 1, z: 2)),
            draw(target: output, slice: 0, sampled: input, colors: firstColors),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1, depth: 3),
            .read(buffer: read, byteCount: 520),
            .renderIndexed(pipeline: pipeline, target: texture, colorLoad: .load,
                           indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3),
                           textures: [.init(binding: 0): input], targetMipLevel: 1, targetDepthSlice: 1,
                           additionalColors: [2: .init(texture: output, depthSlice: 0)],
                           raster: .init(scissor: .init(x: 0, y: 0, width: 1, height: 1))),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1, depth: 3),
            .read(buffer: read, byteCount: 520),
        ])
        let snapshots = try await ticket.wait()
        let planes = snapshots.map { words in [0, 1, 64, 65, 128, 129].map { words[$0] } }
        #expect(planes == [[0xff, 0xff, 0xff00ff40, 0xff00ff40, 0xff00ffff, 0xff00ffff],
                           [0xffff, 0xff, 0xff00ffff, 0xff00ff40, 0xff00ffff, 0xff00ffff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
