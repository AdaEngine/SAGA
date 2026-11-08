import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

@Test(arguments: [
    (TextureFormat.bc1RGBAUnorm, GPUFeature.textureCompressionBC, GPUFeature.textureCompressionBCSliced3D, UInt32(4)),
    (.astc5x4Unorm, .textureCompressionASTC, .textureCompressionASTCSliced3D, 5),
    (.astc5x4HDR, .textureCompressionASTCHDR, .textureCompressionASTCSliced3D, 5),
])
func compressedVolumesRequireFormatAndSliced3DFeatures(
    format: TextureFormat, base: GPUFeature, sliced: GPUFeature, blockWidth: UInt32
) throws {
    let volume = TextureDescriptor(width: 3 * blockWidth, height: 12, format: format,
                                   usage: [.textureBinding, .copyDestination], mipLevelCount: 3, dimension: .d3, depth: 6)
    try volume.validate(against: .init(maxBufferSize: 1024, features: [base, sliced]))
    for features: Set<GPUFeature> in [[base], [sliced], [.textureCompressionBC, .textureCompressionASTC, .textureCompressionASTCHDR]] {
        #expect(throws: GPUError.validation("Feature not advertised")) {
            try volume.validate(against: .init(maxBufferSize: 1024, features: features))
        }
    }
    let etc2 = TextureDescriptor(width: 12, height: 12, format: .etc2RGB8Unorm,
                                 usage: .textureBinding, dimension: .d3, depth: 6)
    #expect(throws: GPUError.self) {
        try etc2.validate(against: .init(maxBufferSize: 1024, features: Set(GPUFeature.allCases)))
    }
}

#if canImport(Metal)
private let compressedVolumeReadShader = """
    @group(0) @binding(0) var input: texture_3d<f32>;
    @group(0) @binding(1) var copied: texture_3d<f32>;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(copied, vec3u(5u, 4u, 1u), 0u);
        let b = textureLoad(copied, vec3u(5u, 4u, 2u), 0u);
        let c = textureLoad(input, vec3u(5u, 4u, 2u), 0u);
        out = vec4f(a.x, b.x, c.x, a.w);
    }
    """

@Test(arguments: [
    (TextureFormat.bc1RGBAUnorm, GPUFeature.textureCompressionBC, GPUFeature.textureCompressionBCSliced3D),
    (.astc5x4Unorm, .textureCompressionASTC, .textureCompressionASTCSliced3D),
    (.astc5x4HDR, .textureCompressionASTCHDR, .textureCompressionASTCSliced3D),
])
func compressedVolumeCopiesPreserveBlockRowsAndDepthOnMetal(format: TextureFormat, base: GPUFeature, sliced: GPUFeature) async throws {
    let backend = try MetalBackend.makeDevice()
    let features: Set<GPUFeature> = [base, sliced]
    guard features.isSubset(of: backend.capabilities.features) else { return }
    let device = try await Device.make(backend: backend, requiredFeatures: features)
    let bc = format == .bc1RGBAUnorm
    let hdr = format == .astc5x4HDR
    let blockWidth: UInt32 = bc ? 4 : 5
    let blocks: [[UInt32]]
    if bc {
        // Equal RGB565 endpoints and zero selectors.
        blocks = [[0x40004000, 0], [0x80008000, 0], [0xc000c000, 0]]
    } else if hdr {
        // Constant-color ASTC blocks with red endpoints 0.5, 1.5 and 2.0.
        blocks = [0x3800, 0x3e00, 0x4000].map { [0xfffffffc, 0xffffffff, $0, 0x3c000000] }
    } else {
        blocks = [0x4040, 0x8080, 0xc0c0].map { [0xfffffdfc, 0xffffffff, $0, 0xffff0000] }
    }
    let expected: [Float] = hdr ? [0.5, 1.5, 2] : [0.25, 0.5, 0.75]
    let blockBytes = blocks[0].count * 4
    let stride = blockBytes + 3
    let bytes = [UInt8(0xee)] + blocks[0].withUnsafeBytes { Array($0) } +
        [UInt8](repeating: 0xee, count: 2 * stride - blockBytes) + blocks[1].withUnsafeBytes { Array($0) }
    let texture = TextureDescriptor(width: blockWidth * 3, height: 12, format: format,
                                    usage: [.copySource, .copyDestination, .textureBinding],
                                    mipLevelCount: 3, dimension: .d3, depth: 6)
    let input = try await device.createTexture(texture)
    let copied = try await device.createTexture(texture)
    let inputView = try await device.createTextureView(texture: input, .init(baseMipLevel: 1, mipLevelCount: 1))
    let copiedView = try await device.createTextureView(texture: copied, .init(baseMipLevel: 1, mipLevelCount: 1))
    let staging = try await device.createBuffer(.init(size: UInt64(blockBytes), usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readbackSize = UInt64(1280 + 2 * blockBytes)
    let readback = try await device.createBuffer(.init(size: readbackSize, usage: [.copyDestination, .mapRead]))
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(compressedVolumeReadShader), to: .msl)
    let pipeline = try await device.createComputePipeline(shader)
    let pair = TextureCopyRegion(x: blockWidth, y: 4, width: blockWidth, height: 4, mipLevel: 1, depth: 2)
    let commands: [BufferCommand] = [
        .writeTexture(destination: input, data: bytes, layout: .init(offset: 1, bytesPerRow: UInt32(stride), rowsPerImage: 2),
                      region: pair),
        .write(buffer: staging, offset: 0, words: blocks[2]),
        .copyBufferToTexture(source: staging, layout: .init(), destination: input,
                             region: .init(x: blockWidth, y: 4, width: blockWidth, height: 4, mipLevel: 1, z: 2)),
        .copyTextureToTexture(source: input, destination: copied, sourceRegion: pair,
                              destinationOrigin: .init(x: blockWidth, y: 4, mipLevel: 1, z: 1)),
        .copyTextureToBuffer(source: copied, destination: readback, bytesPerRow: 256, mipLevel: 1, depth: 3),
        .read(buffer: readback, byteCount: readbackSize),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): inputView, .init(binding: 1): copiedView]),
        .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]
    let snapshots = try await device.submitCommands(commands).wait()
    try #require(snapshots.map(\.count) == [Int(readbackSize / 4), 4])
    let words = blocks[0].count
    for (plane, block) in [[UInt32](repeating: 0, count: words), blocks[0], blocks[1]].enumerated() {
        let top = plane * 128
        let bottom = top + 64
        #expect(Array(snapshots[0][top..<(top + 2 * words)]) == Array(repeating: 0, count: 2 * words))
        #expect(Array(snapshots[0][bottom..<(bottom + words)]) == Array(repeating: 0, count: words))
        #expect(Array(snapshots[0][(bottom + words)..<(bottom + 2 * words)]) == block)
    }
    let pixels = snapshots[1].map(Float.init(bitPattern:))
    for (actual, reference) in zip(pixels, expected + [1]) {
        #expect(abs(actual - reference) < 0.04)
    }
    await device.close()
}
#endif
