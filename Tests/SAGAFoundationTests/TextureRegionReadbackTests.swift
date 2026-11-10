import Testing
import SAGATypes
import SAGACore
import SAGAMetal

@Test func validatesTextureRegionReadback() throws {
    let image = ResourceID(device: 1, serial: 1)
    let output = ResourceID(device: 1, serial: 2)
    let texture = TextureDescriptor(width: 8, height: 8, format: .rgba8Unorm,
        usage: .copySource, arrayLayerCount: 3)
    let buffers = [output: BufferDescriptor(size: 1024, usage: .copyDestination)]
    func check(_ region: TextureCopyRegion) throws {
        try BufferCommand.copyTextureRegionToBuffer(source: image, destination: output,
            layout: .init(offset: 4, bytesPerRow: 256, rowsPerImage: 2), region: region)
            .validate(device: 1, descriptors: buffers, textures: [image: texture])
    }
    try check(.init(x: 2, y: 3, width: 2, height: 2, arrayLayer: 1, depth: 2))
    #expect(throws: GPUError.self) { try check(.init(x: 7, width: 2, height: 1)) }
    #expect(throws: GPUError.self) { try check(.init(width: 1, height: 1, arrayLayer: 2, depth: 2)) }
    let depth = TextureDescriptor(width: 8, height: 8, format: .depth32Float, usage: .copySource)
    #expect(throws: GPUError.self) {
        try BufferCommand.copyTextureRegionToBuffer(source: image, destination: output, layout: .init(bytesPerRow: 256),
            region: .init(x: 1, width: 2, height: 2))
            .validate(device: 1, descriptors: buffers, textures: [image: depth])
    }
}

#if canImport(Metal)
@Test func readsTextureRegionsAcrossArrayLayersAndVolumeSlicesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        for dimension: TextureDimension in [.d2, .d3] {
            let texture = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
                usage: [.copySource, .copyDestination], arrayLayerCount: dimension == .d2 ? 3 : 1,
                dimension: dimension, depth: dimension == .d3 ? 3 : 1))
            let output = try await device.createBuffer(.init(size: 528, usage: [.copyDestination, .mapRead]))
            let full = TextureCopyRegion(width: 4, height: 2, depth: 3)
            let bytes = (0..<96).map { UInt8($0) }
            let region = TextureCopyRegion(x: 1, y: 1, width: 2, height: 1,
                arrayLayer: dimension == .d2 ? 1 : 0, z: dimension == .d3 ? 1 : 0, depth: 2)
            let result = try await device.submitCommands([
                .writeTexture(destination: texture, data: bytes, layout: .init(bytesPerRow: 16, rowsPerImage: 2), region: full),
                .write(buffer: output, offset: 0, words: [0xdeadbeef, 0, 0, 0xdeadbeef]),
                .copyTextureRegionToBuffer(source: texture, destination: output,
                    layout: .init(offset: 4, bytesPerRow: 256, rowsPerImage: 1), region: region),
                .read(buffer: output, byteCount: 268),
            ]).wait()
            #expect(result[0][0] == 0xdeadbeef && result[0][3] == 0xdeadbeef)
            #expect([result[0][1], result[0][2], result[0][65], result[0][66]] ==
                [0x37363534, 0x3b3a3938, 0x57565554, 0x5b5a5958])
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

#if canImport(Metal)
@Test func readsCompressedTextureRegionOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.textureCompressionBC])
    do {
        let image = try await device.createTexture(.init(width: 8, height: 8, format: .bc1RGBAUnorm,
            usage: [.copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .writeTexture(destination: image, data: (0..<32).map { UInt8($0) },
                layout: .init(bytesPerRow: 16), region: .init(width: 8, height: 8)),
            .copyTextureRegionToBuffer(source: image, destination: output, layout: .init(),
                region: .init(x: 4, y: 4, width: 4, height: 4)),
            .read(buffer: output, byteCount: 8),
        ]).wait()
        #expect(result == [[0x1b1a1918, 0x1f1e1d1c]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
