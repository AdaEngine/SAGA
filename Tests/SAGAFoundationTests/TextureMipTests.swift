import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let mipShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 7>;
    @group(0) @binding(1) var image: texture_2d<f32>;
    @group(0) @binding(2) var filter: sampler;
    @compute @workgroup_size(1) fn run() {
        out[0] = textureLoad(image, vec2u(4, 2), 0u);
        out[1] = textureLoad(image, vec2u(1, 0), 1u);
        out[2] = textureLoad(image, vec2u(0, 0), 2u);
        out[3] = textureSampleLevel(image, filter, vec2f(0.75, 0.5), 0.5);
        out[4] = textureLoad(image, vec2u(0, 0), 1u);
        let size = vec2f(textureDimensions(image, 1u));
        out[5] = vec4f(f32(textureNumLevels(image)), size.x, size.y, f32(i32(textureDimensions(image, 2u).x)));
        out[6] = textureLoad(image, vec2u(2, 0), 1u);
    }
    """

@Test func validatesTextureMipCountsAndCopyFootprints() throws {
    let caps = try DeviceCapabilities(maxBufferSize: 4096)
    let usage: TextureUsage = [.copySource, .copyDestination]
    func texture(_ levels: UInt32) -> TextureDescriptor {
        .init(width: 5, height: 3, format: .rgba8Unorm, usage: usage, mipLevelCount: levels)
    }
    #expect(TextureDescriptor(width: 5, height: 3, format: .rgba8Unorm, usage: usage).mipLevelCount == 1)
    try texture(2).validate(against: caps)
    try texture(3).validate(against: caps)
    for count: UInt32 in [0, 4, .max] {
        #expect(throws: GPUError.self) { try texture(count).validate(against: caps) }
    }

    let buffer = ResourceID(device: 1, serial: 1)
    let image = ResourceID(device: 1, serial: 2)
    let buffers = [buffer: BufferDescriptor(size: 8, usage: [.copySource, .copyDestination])]
    let textures = [image: texture(3)]
    try BufferCommand.copyTextureToBuffer(source: image, destination: buffer, bytesPerRow: 256, mipLevel: 1)
        .validate(device: 1, descriptors: buffers, textures: textures)
    for level: UInt32 in [0, 3, .max] {
        #expect(throws: GPUError.self) {
            try BufferCommand.copyTextureToBuffer(source: image, destination: buffer, bytesPerRow: 256, mipLevel: level)
                .validate(device: 1, descriptors: buffers, textures: textures)
        }
    }
    func upload(_ region: TextureCopyRegion) -> BufferCommand {
        .copyBufferToTexture(source: buffer, layout: .init(), destination: image, region: region)
    }
    try upload(.init(width: 2, height: 1, mipLevel: 1))
        .validate(device: 1, descriptors: buffers, textures: textures)
    for region in [
        TextureCopyRegion(x: 2, width: 1, height: 1, mipLevel: 1),
        .init(width: 2, height: 1, mipLevel: 2),
        .init(width: 0, height: 0, mipLevel: 3),
    ] {
        #expect(throws: GPUError.self) {
            try upload(region).validate(device: 1, descriptors: buffers, textures: textures)
        }
    }
}

#if canImport(Metal)
@Test func preservesIndependentMipLevelsAcrossUploadsSamplingAndRenderOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let shader = try compiler.compileCompute(.wgsl(mipShader), to: .msl)
        let compute = try await device.createComputePipeline(shader)
        let clearShader = ShaderInput.wgsl("""
            @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 0, 1, 1); }
            """)
        let render = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(clearShader, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(clearShader, entryPoint: "fs", to: .msl), colorFormat: .rgba8Unorm)
        let image = try await device.createTexture(.init(width: 5, height: 3, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination, .textureBinding, .renderAttachment], mipLevelCount: 3))
        let upload = try await device.createBuffer(.init(size: 536, usage: [.copySource, .copyDestination]))
        let out = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
        let filter = try await device.createSampler(.init(mipmapFilter: .linear))
        let dispatch = BufferCommand.dispatchCompute(
            pipeline: compute, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
            textures: [.init(binding: 1): image], samplers: [.init(binding: 2): filter])
        let red = Array(repeating: UInt32(0xff0000ff), count: 5)
        let green: UInt32 = 0xff00ff00
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([
                .render(pipeline: render, target: image, clear: blue, vertexCount: 0, targetMipLevel: 3),
            ])
        }
        let first = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: red),
            .write(buffer: upload, offset: 256, words: red),
            .write(buffer: upload, offset: 512, words: red + [green]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: image,
                                 region: .init(width: 5, height: 3)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 532), destination: image,
                                 region: .init(x: 1, width: 1, height: 1, mipLevel: 1)),
            dispatch,
            .copy(source: out, destination: read, byteCount: 112),
            .read(buffer: read, byteCount: 112),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 8),
        ])
        let initial: [Float] = [
            1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0, 1,
            0, 0, 0, 0, 3, 2, 1, 1, 0, 0, 0, 0,
        ]
        #expect(try await first.wait() == [initial.map(\.bitPattern), [0, green]])
        let second = try await device.submitCommands([
            .render(pipeline: render, target: image, clear: blue, vertexCount: 0, targetMipLevel: 2),
            dispatch,
            .copy(source: out, destination: read, byteCount: 112),
            .read(buffer: read, byteCount: 112),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 2),
            .read(buffer: read, byteCount: 4),
        ])
        var rendered = initial
        rendered.replaceSubrange(8..<12, with: [0, 0, 1, 1])
        #expect(try await second.wait() == [rendered.map(\.bitPattern), [0xffff0000]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
