import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let astcShapes: [(width: UInt32, height: UInt32, linear: TextureFormat, srgb: TextureFormat, hdr: TextureFormat)] = [
    (4, 4, .astc4x4Unorm, .astc4x4UnormSrgb, .astc4x4HDR),
    (5, 4, .astc5x4Unorm, .astc5x4UnormSrgb, .astc5x4HDR),
    (5, 5, .astc5x5Unorm, .astc5x5UnormSrgb, .astc5x5HDR),
    (6, 5, .astc6x5Unorm, .astc6x5UnormSrgb, .astc6x5HDR),
    (6, 6, .astc6x6Unorm, .astc6x6UnormSrgb, .astc6x6HDR),
    (8, 5, .astc8x5Unorm, .astc8x5UnormSrgb, .astc8x5HDR),
    (8, 6, .astc8x6Unorm, .astc8x6UnormSrgb, .astc8x6HDR),
    (8, 8, .astc8x8Unorm, .astc8x8UnormSrgb, .astc8x8HDR),
    (10, 5, .astc10x5Unorm, .astc10x5UnormSrgb, .astc10x5HDR),
    (10, 6, .astc10x6Unorm, .astc10x6UnormSrgb, .astc10x6HDR),
    (10, 8, .astc10x8Unorm, .astc10x8UnormSrgb, .astc10x8HDR),
    (10, 10, .astc10x10Unorm, .astc10x10UnormSrgb, .astc10x10HDR),
    (12, 10, .astc12x10Unorm, .astc12x10UnormSrgb, .astc12x10HDR),
    (12, 12, .astc12x12Unorm, .astc12x12UnormSrgb, .astc12x12HDR),
]

@Test(arguments: astcShapes)
func astcDimensionsControlAlignmentAndCopyFootprint(
    width: UInt32, height: UInt32, linear: TextureFormat, srgb: TextureFormat, hdr: TextureFormat
) throws {
    let ldrCaps = try DeviceCapabilities(maxBufferSize: 512, features: [.textureCompressionASTC])
    let hdrCaps = try DeviceCapabilities(maxBufferSize: 512, features: [.textureCompressionASTCHDR])
    let source = ResourceID(device: 1, serial: 1)
    let target = ResourceID(device: 1, serial: 2)
    let buffer = ResourceID(device: 1, serial: 3)
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let baseWidth = width * 3
    let baseHeight = height * 3
    let tail = TextureCopyRegion(x: width, y: height, width: width, height: height, mipLevel: 1)
    func descriptor(_ format: TextureFormat) -> TextureDescriptor {
        .init(width: baseWidth, height: baseHeight, format: format, usage: usage, mipLevelCount: 2)
    }
    for (format, capabilities) in [(linear, ldrCaps), (srgb, ldrCaps), (hdr, hdrCaps)] {
        let image = descriptor(format)
        try image.validate(against: capabilities)
        #expect(throws: GPUError.validation("Feature not advertised")) {
            try image.validate(against: format == hdr ? ldrCaps : hdrCaps)
        }
        for (w, h) in [(baseWidth + 1, baseHeight), (baseWidth, baseHeight + 1)] {
            #expect(throws: GPUError.self) {
                try TextureDescriptor(width: w, height: h, format: format, usage: usage).validate(against: capabilities)
            }
        }
        let textures = [source: image]
        let read = BufferCommand.copyTextureToBuffer(source: source, destination: buffer, bytesPerRow: 256, mipLevel: 1)
        try read.validate(device: 1, descriptors: [buffer: .init(size: 288, usage: .copyDestination)], textures: textures)
        #expect(throws: GPUError.self) {
            try read.validate(device: 1, descriptors: [buffer: .init(size: 287, usage: .copyDestination)], textures: textures)
        }
        let block = [UInt8](repeating: 0, count: 16)
        try BufferCommand.writeTexture(destination: source, data: block, region: tail)
            .validate(device: 1, descriptors: [:], textures: textures)
        #expect(throws: GPUError.self) {
            try BufferCommand.writeTexture(destination: source, data: block,
                    region: .init(x: width - 1, y: height, width: width, height: height, mipLevel: 1))
                .validate(device: 1, descriptors: [:], textures: textures)
        }
    }
    let copy = BufferCommand.copyTextureToTexture(source: source, destination: target, sourceRegion: tail,
                                                 destinationOrigin: .init(x: width, y: height, mipLevel: 1))
    try copy.validate(device: 1, descriptors: [:], textures: [source: descriptor(linear), target: descriptor(srgb)])
    #expect(throws: GPUError.self) {
        try copy.validate(device: 1, descriptors: [:], textures: [source: descriptor(linear), target: descriptor(hdr)])
    }
}

#if canImport(Metal)
private let astcReadShader = """
    @group(0) @binding(0) var linear: texture_2d<f32>;
    @group(0) @binding(1) var srgb: texture_2d<f32>;
    @group(0) @binding(2) var wide: texture_2d<f32>;
    @group(0) @binding(3) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(linear, vec2u(6u, 5u), 0u);
        let b = textureLoad(srgb, vec2u(6u, 5u), 0u);
        let c = textureLoad(wide, vec2u(4u, 3u), 0u);
        out = vec4f(a.x, b.x, c.x, c.w);
    }
    """

@Test(arguments: [false, true])
func astcRectangularMipCopiesAndShaderDecodeOnMetal(hdr: Bool) async throws {
    let backend = try MetalBackend.makeDevice()
    let features: Set<GPUFeature> = hdr ? [.textureCompressionASTC, .textureCompressionASTCHDR] : [.textureCompressionASTC]
    guard features.isSubset(of: backend.capabilities.features) else { return }
    let device = try await Device.make(backend: backend, requiredFeatures: features)
    for shape in astcShapes {
        for format in hdr ? [shape.hdr] : [shape.linear, shape.srgb] {
            let image = try await device.createTexture(.init(width: shape.width, height: shape.height, format: format,
                                                            usage: .textureBinding))
            try await device.destroyTexture(image)
        }
    }
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let linear = try await device.createTexture(.init(width: 15, height: 12, format: .astc5x4Unorm, usage: usage, mipLevelCount: 2))
    let srgb = try await device.createTexture(.init(width: 15, height: 12, format: .astc5x4UnormSrgb, usage: usage, mipLevelCount: 2))
    let wide = try await device.createTexture(.init(width: 20, height: 16, format: hdr ? .astc10x8HDR : .astc10x8Unorm,
                                                   usage: usage, mipLevelCount: 3))
    let linearView = try await device.createTextureView(texture: linear, .init(baseMipLevel: 1, mipLevelCount: 1))
    let srgbView = try await device.createTextureView(texture: srgb, .init(baseMipLevel: 1, mipLevelCount: 1))
    let wideView = try await device.createTextureView(texture: wide, .init(baseMipLevel: 2, mipLevelCount: 1))
    let source = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 288, usage: [.mapRead, .copyDestination]))
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(astcReadShader), to: .msl))
    // Void-extent blocks with all coordinates set to one encode constant colors.
    let gray: [UInt8] = [0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xff, 0xff]
    let wideWords: [UInt32] = hdr ? [0xfffffffc, 0xffffffff, 0x00003e00, 0x3c000000] :
        [0xfffffdfc, 0xffffffff, 0x0000bfbf, 0xffff0000]
    let corner = TextureCopyRegion(x: 5, y: 4, width: 5, height: 4, mipLevel: 1)
    let commands: [BufferCommand] = [
        .writeTexture(destination: linear, data: [0xee] + gray, layout: .init(offset: 1), region: corner),
        .copyTextureToTexture(source: linear, destination: srgb, sourceRegion: corner,
                              destinationOrigin: .init(x: 5, y: 4, mipLevel: 1)),
        .copyTextureToBuffer(source: srgb, destination: readback, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: readback, byteCount: 288),
        .write(buffer: source, offset: 0, words: wideWords),
        .copyBufferToTexture(source: source, layout: .init(), destination: wide, region: .init(width: 10, height: 8, mipLevel: 2)),
        .copyTextureToBuffer(source: wide, destination: readback, bytesPerRow: 256, mipLevel: 2),
        .read(buffer: readback, byteCount: 16),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 3): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): linearView, .init(binding: 1): srgbView, .init(binding: 2): wideView]),
        .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]
    let snapshots = try await device.submitCommands(commands).wait()
    try #require(snapshots.map(\.count) == [72, 4, 4])
    #expect(Array(snapshots[0][0..<8]) == Array(repeating: 0, count: 8))
    #expect(Array(snapshots[0][64..<72]) == [0, 0, 0, 0, 0xfffffdfc, 0xffffffff, 0x80808080, 0xffff8080])
    #expect(snapshots[1] == wideWords)
    let pixels = snapshots[2].map(Float.init(bitPattern:))
    #expect((0.49...0.51).contains(pixels[0]))
    #expect((0.20...0.23).contains(pixels[1]))
    #expect(hdr ? pixels[2] == 1.5 : (0.74...0.76).contains(pixels[2]))
    #expect(pixels[3] == 1)
    await device.close()
}
#endif
