import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let etcFormats: [(TextureFormat, UInt64)] = [
    (.etc2RGB8Unorm, 8), (.etc2RGB8UnormSrgb, 8), (.etc2RGB8A1Unorm, 8), (.etc2RGB8A1UnormSrgb, 8),
    (.etc2RGBA8Unorm, 16), (.etc2RGBA8UnormSrgb, 16),
    (.eacR11Unorm, 8), (.eacR11Snorm, 8), (.eacRG11Unorm, 16), (.eacRG11Snorm, 16),
]

@Test(arguments: etcFormats)
func etc2FormatsRequireTheirOwnFeatureAndUseTheCorrectCopyFootprint(format: TextureFormat, bytes: UInt64) throws {
    let capabilities = try DeviceCapabilities(maxBufferSize: 128, features: [.textureCompressionETC2])
    let onlyBC = try DeviceCapabilities(maxBufferSize: 128, features: [.textureCompressionBC])
    let descriptor = TextureDescriptor(width: 12, height: 8, format: format,
        usage: [.copySource, .copyDestination, .textureBinding], mipLevelCount: 3)
    try descriptor.validate(against: capabilities)
    #expect(throws: GPUError.validation("Feature not advertised")) { try descriptor.validate(against: onlyBC) }
    for invalid in [
        TextureDescriptor(width: 4, height: 5, format: format, usage: .copyDestination),
        TextureDescriptor(width: 4, height: 4, format: format, usage: .renderAttachment),
        TextureDescriptor(width: 4, height: 4, format: format, usage: .storageBinding),
        TextureDescriptor(width: 4, height: 4, format: format, usage: .textureBinding, dimension: .d3, depth: 4),
    ] {
        #expect(throws: GPUError.self) { try invalid.validate(against: capabilities) }
    }

    let image = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let copy = BufferCommand.copyTextureToBuffer(source: image, destination: buffer, bytesPerRow: 256, mipLevel: 1)
    try copy.validate(device: 1, descriptors: [buffer: .init(size: bytes * 2, usage: .copyDestination)], textures: [image: descriptor])
    #expect(throws: GPUError.self) {
        try copy.validate(device: 1, descriptors: [buffer: .init(size: bytes * 2 - 1, usage: .copyDestination)], textures: [image: descriptor])
    }
}

#if canImport(Metal)
private let etc2ReadShader = """
    @group(0) @binding(0) var linear: texture_2d<f32>;
    @group(0) @binding(1) var srgb: texture_2d<f32>;
    @group(0) @binding(2) var eac: texture_2d<f32>;
    @group(0) @binding(3) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(linear, vec2u(1u), 0u);
        let b = textureLoad(srgb, vec2u(1u), 0u);
        let c = textureLoad(eac, vec2u(1u), 0u);
        out = vec4f(a.x, b.x, c.xy);
    }
    """

@Test func etc2AndEACUploadCopyAndDecodeOnMetal() async throws {
    let backend = try MetalBackend.makeDevice()
    guard backend.capabilities.features.contains(.textureCompressionETC2) else { return }
    let device = try await Device.make(backend: backend, requiredFeatures: [.textureCompressionETC2])
    for (format, _) in etcFormats {
        let image = try await device.createTexture(.init(width: 4, height: 4, format: format, usage: .textureBinding))
        try await device.destroyTexture(image)
    }
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let linear = try await device.createTexture(.init(width: 8, height: 8, format: .etc2RGB8Unorm, usage: usage,
                                                     mipLevelCount: 3, arrayLayerCount: 2))
    let srgb = try await device.createTexture(.init(width: 4, height: 4, format: .etc2RGB8UnormSrgb, usage: usage, mipLevelCount: 2))
    let eac = try await device.createTexture(.init(width: 4, height: 4, format: .eacRG11Unorm, usage: usage))
    let linearView = try await device.createTextureView(texture: linear, .init(baseMipLevel: 2, mipLevelCount: 1,
                                                                              dimension: .d2, arrayLayerCount: 1))
    let srgbView = try await device.createTextureView(texture: srgb, .init(baseMipLevel: 1, mipLevelCount: 1))
    let source = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let result = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(etc2ReadShader), to: .msl))
    // ETC2 individual mode with the same 4-bit RGB base in both subblocks and zero selectors.
    let gray: [UInt8] = [0x66, 0x66, 0x66, 0, 0, 0, 0, 0]
    let bright: [UInt8] = [0xaa, 0xaa, 0xaa, 0, 0, 0, 0, 0]
    let data = [UInt8(0xee)] + gray + Array(repeating: 0xee, count: 14) + bright
    let commands: [BufferCommand] = [
        .writeTexture(destination: linear, data: data, layout: .init(offset: 1, bytesPerRow: 11, rowsPerImage: 2),
                      region: .init(width: 4, height: 4, mipLevel: 2, depth: 2)),
        .copyTextureToBuffer(source: linear, destination: readback, bytesPerRow: 256, mipLevel: 2, arrayLayer: 1),
        .read(buffer: readback, byteCount: 8),
        .copyTextureToTexture(source: linear, destination: srgb, sourceRegion: .init(width: 4, height: 4, mipLevel: 2),
                              destinationOrigin: .init(mipLevel: 1)),
        .copyTextureToBuffer(source: srgb, destination: readback, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: readback, byteCount: 8),
        .write(buffer: source, offset: 0, words: [0x80, 0, 0x40, 0]),
        .copyBufferToTexture(source: source, layout: .init(), destination: eac, region: .init(width: 4, height: 4)),
        .copyTextureToBuffer(source: eac, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 16),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 3): result], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): linearView, .init(binding: 1): srgbView, .init(binding: 2): eac]),
        .copy(source: result, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]
    let snapshots = try await device.submitCommands(commands).wait()
    try #require(snapshots.map(\.count) == [2, 2, 4, 4])
    #expect(snapshots[0] == [0x00aaaaaa, 0])
    #expect(snapshots[1] == [0x00666666, 0])
    #expect(snapshots[2] == [0x80, 0, 0x40, 0])
    let pixels = snapshots[3].map(Float.init(bitPattern:))
    #expect((0.40...0.42).contains(pixels[0]))
    #expect((0.13...0.15).contains(pixels[1]))
    #expect((0.48...0.52).contains(pixels[2]))
    #expect((0.23...0.27).contains(pixels[3]))
    await device.close()
}
#endif
