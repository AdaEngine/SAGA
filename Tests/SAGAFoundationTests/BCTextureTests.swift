import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let bcFormats: [TextureFormat] = [
    .bc1RGBAUnorm, .bc1RGBAUnormSrgb, .bc2RGBAUnorm, .bc2RGBAUnormSrgb, .bc3RGBAUnorm, .bc3RGBAUnormSrgb,
    .bc4RUnorm, .bc4RSnorm, .bc5RGUnorm, .bc5RGSnorm, .bc6HRGBUfloat, .bc6HRGBFloat, .bc7RGBAUnorm, .bc7RGBAUnormSrgb,
]

@Test(arguments: bcFormats)
func bcFormatsRequireFeatureAndAligned2DAllocations(format: TextureFormat) throws {
    let available = try DeviceCapabilities(maxBufferSize: 1 << 20, features: [.textureCompressionBC])
    let unavailable = try DeviceCapabilities(maxBufferSize: 1 << 20)
    let descriptor = TextureDescriptor(width: 8, height: 4, format: format,
        usage: [.textureBinding, .copySource, .copyDestination], mipLevelCount: 3, arrayLayerCount: 2)
    try descriptor.validate(against: available)
    #expect(throws: GPUError.validation("Feature not advertised")) { try descriptor.validate(against: unavailable) }
    for invalid in [
        TextureDescriptor(width: 5, height: 4, format: format, usage: .copyDestination),
        TextureDescriptor(width: 4, height: 4, format: format, usage: .renderAttachment),
        TextureDescriptor(width: 4, height: 4, format: format, usage: .storageBinding),
        TextureDescriptor(width: 4, height: 4, format: format, usage: .textureBinding, dimension: .d3, depth: 4),
    ] {
        #expect(throws: GPUError.self) { try invalid.validate(against: available) }
    }
}

@Test func bcCopiesValidatePhysicalBlocksAndTheirByteFootprints() throws {
    let image = ResourceID(device: 1, serial: 1)
    let srgb = ResourceID(device: 1, serial: 2)
    let bc3 = ResourceID(device: 1, serial: 3)
    let buffer = ResourceID(device: 1, serial: 4)
    let textures: [ResourceID: TextureDescriptor] = [
        image: .init(width: 8, height: 8, format: .bc1RGBAUnorm, usage: [.copySource, .copyDestination], mipLevelCount: 3),
        srgb: .init(width: 4, height: 4, format: .bc1RGBAUnormSrgb, usage: [.copySource, .copyDestination], mipLevelCount: 2),
        bc3: .init(width: 4, height: 4, format: .bc3RGBAUnorm, usage: [.copySource, .copyDestination]),
    ]
    let buffers: [ResourceID: BufferDescriptor] = [buffer: .init(size: 24, usage: [.copySource, .copyDestination])]
    let tail = TextureCopyRegion(width: 4, height: 4, mipLevel: 2)
    func check(_ command: BufferCommand) throws {
        try command.validate(device: 1, descriptors: buffers, textures: textures)
    }
    try check(.writeTexture(destination: image, data: Array(repeating: 0, count: 9), layout: .init(offset: 1), region: tail))
    try check(.copyTextureToTexture(source: image, destination: srgb, sourceRegion: tail, destinationOrigin: .init(mipLevel: 1)))
    try check(.copyTextureToBuffer(source: image, destination: buffer, bytesPerRow: 256, mipLevel: 2, destinationOffset: 16))
    try check(.copyBufferToTexture(source: buffer, layout: .init(), destination: bc3, region: .init(width: 4, height: 4)))
    for invalid in [
        .writeTexture(destination: image, data: Array(repeating: 0, count: 7), region: tail),
        .writeTexture(destination: image, data: Array(repeating: 0, count: 8),
                      region: .init(width: 2, height: 2, mipLevel: 2)),
        .writeTexture(destination: image, data: Array(repeating: 0, count: 8), region: .init(x: 2, width: 4, height: 4)),
        .copyTextureToTexture(source: image, destination: bc3, sourceRegion: tail),
        .copyTextureToBuffer(source: image, destination: buffer, bytesPerRow: 256, mipLevel: 2, destinationOffset: 4),
        .copyBufferToTexture(source: buffer, layout: .init(offset: 8), destination: bc3, region: .init(width: 4, height: 4)),
        .copyBufferToTexture(source: buffer, layout: .init(offset: 16), destination: bc3, region: .init(width: 4, height: 4)),
    ] as [BufferCommand] {
        #expect(throws: GPUError.self) { try check(invalid) }
    }
}

private let bcSampleShader = """
    @group(0) @binding(0) var linear: texture_2d<f32>;
    @group(0) @binding(1) var srgb: texture_2d<f32>;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(linear, vec2u(1u), 0u);
        let b = textureLoad(srgb, vec2u(1u), 0u);
        out = vec4f(a.x, b.x, a.w, b.w);
    }
    """

#if canImport(Metal)
@Test func uploadsCopiesAndSamplesBCBlocksAndTailMipsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.textureCompressionBC])
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let linear = try await device.createTexture(.init(width: 8, height: 8, format: .bc1RGBAUnorm, usage: usage, mipLevelCount: 3))
    let srgb = try await device.createTexture(.init(width: 4, height: 4, format: .bc1RGBAUnormSrgb, usage: usage, mipLevelCount: 2))
    let bc3 = try await device.createTexture(.init(width: 4, height: 4, format: .bc3RGBAUnorm, usage: usage))
    let linearView = try await device.createTextureView(texture: linear, .init(baseMipLevel: 2, mipLevelCount: 1))
    let srgbView = try await device.createTextureView(texture: srgb, .init(baseMipLevel: 1, mipLevelCount: 1))
    let source = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let result = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 272, usage: [.mapRead, .copyDestination]))
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(bcSampleShader), to: .msl))
    // Equal endpoints decode to a neutral mid-gray. Selector bits are zero.
    let gray: [UInt8] = [0xef, 0x7b, 0xef, 0x7b, 0, 0, 0, 0]
    let tail = TextureCopyRegion(width: 4, height: 4, mipLevel: 2)
    let commands: [BufferCommand] = [
        .copyTextureToBuffer(source: linear, destination: readback, bytesPerRow: 256, mipLevel: 2),
        .read(buffer: readback, byteCount: 8),
        .writeTexture(destination: linear, data: gray, region: .init(x: 4, y: 4, width: 4, height: 4)),
        .copyTextureToBuffer(source: linear, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 272),
        .writeTexture(destination: linear, data: [0xee] + gray, layout: .init(offset: 1), region: tail),
        .copyTextureToTexture(source: linear, destination: srgb, sourceRegion: tail, destinationOrigin: .init(mipLevel: 1)),
        .copyTextureToBuffer(source: srgb, destination: readback, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: readback, byteCount: 8),
        .write(buffer: source, offset: 0, words: [0x1234ffff, 0, 0x7bef7bef, 0]),
        .copyBufferToTexture(source: source, layout: .init(), destination: bc3, region: .init(width: 4, height: 4)),
        .copyTextureToBuffer(source: bc3, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 16),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): result], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): linearView, .init(binding: 1): srgbView]),
        .copy(source: result, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]
    let snapshots = try await device.submitCommands(commands).wait()
    try #require(snapshots.map(\.count) == [2, 68, 2, 4, 4])
    #expect(snapshots[0] == [0, 0])
    #expect(Array(snapshots[1][0..<4]) == [0, 0, 0, 0])
    #expect(Array(snapshots[1][64..<68]) == [0, 0, 0x7bef7bef, 0])
    #expect(snapshots[2] == [0x7bef7bef, 0])
    #expect(snapshots[3] == [0x1234ffff, 0, 0x7bef7bef, 0])
    let pixels = snapshots[4].map(Float.init(bitPattern:))
    #expect(pixels[0] > 0.46 && pixels[0] < 0.50)
    #expect(pixels[1] > 0.17 && pixels[1] < 0.22)
    #expect(pixels[2] == 1 && pixels[3] == 1)
    await device.close()
}
#endif
