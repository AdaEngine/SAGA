import Testing
import SAGATypes
import SAGACore
import SAGAMetal

@Test func mixedDimensionTextureCopyValidatesEachTextureExtent() throws {
    let array = ResourceID(device: 1, serial: 1)
    let volume = ResourceID(device: 1, serial: 2)
    let usage: TextureUsage = [.copySource, .copyDestination]
    let textures: [ResourceID: TextureDescriptor] = [
        array: .init(width: 2, height: 2, format: .rgba8Unorm, usage: usage, arrayLayerCount: 4),
        volume: .init(width: 2, height: 2, format: .rgba8Unorm, usage: usage, dimension: .d3, depth: 4),
    ]
    try BufferCommand.copyTextureToTexture(source: array, destination: volume,
        sourceRegion: .init(width: 2, height: 2, arrayLayer: 1, depth: 2),
        destinationOrigin: .init(z: 1)).validate(device: 1, descriptors: [:], textures: textures)
    try BufferCommand.copyTextureToTexture(source: volume, destination: array,
        sourceRegion: .init(width: 2, height: 2, z: 1, depth: 2),
        destinationOrigin: .init(arrayLayer: 1)).validate(device: 1, descriptors: [:], textures: textures)
    #expect(throws: GPUError.self) {
        try BufferCommand.copyTextureToTexture(source: array, destination: volume,
            sourceRegion: .init(width: 2, height: 2, arrayLayer: 3, depth: 2),
            destinationOrigin: .init(z: 1)).validate(device: 1, descriptors: [:], textures: textures)
    }
    #expect(throws: GPUError.self) {
        try BufferCommand.copyTextureToTexture(source: volume, destination: array,
            sourceRegion: .init(width: 2, height: 2, z: 1, depth: 2),
            destinationOrigin: .init(arrayLayer: 3)).validate(device: 1, descriptors: [:], textures: textures)
    }
}

#if canImport(Metal)
@Test func compressedTextureCopyUsesPhysicalTailMipExtentsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.textureCompressionBC])
    do {
        let usage: TextureUsage = [.copySource, .copyDestination]
        let small = try await device.createTexture(.init(width: 4, height: 4, format: .bc1RGBAUnorm,
            usage: usage, mipLevelCount: 3))
        let large = try await device.createTexture(.init(width: 4, height: 4, format: .bc1RGBAUnorm,
            usage: usage, mipLevelCount: 2, arrayLayerCount: 2))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let block: [UInt8] = [0xef, 0x7b, 0xef, 0x7b, 0, 0, 0, 0]
        let snapshots = try await device.submitCommands([
            .writeTexture(destination: small, data: block, region: .init(width: 4, height: 4, mipLevel: 2)),
            .copyTextureToTexture(source: small, destination: large,
                sourceRegion: .init(width: 4, height: 4, mipLevel: 2),
                destinationOrigin: .init(mipLevel: 1, arrayLayer: 1)),
            .copyTextureToTexture(source: large, destination: small,
                sourceRegion: .init(width: 4, height: 4, mipLevel: 1, arrayLayer: 1),
                destinationOrigin: .init(mipLevel: 2)),
            .copyTextureToBuffer(source: small, destination: read, bytesPerRow: 256, mipLevel: 2),
            .read(buffer: read, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[0x7bef7bef, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func mixedDimensionTextureCopyPreservesArrayLayersAndVolumeSlicesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let usage: TextureUsage = [.copySource, .copyDestination]
        let array = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: usage, arrayLayerCount: 4))
        let volume = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: usage, dimension: .d3, depth: 4))
        let result = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: usage, arrayLayerCount: 4))
        let upload = try await device.createBuffer(.init(size: 260, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 260, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0x11223344]),
            .write(buffer: upload, offset: 256, words: [0xaabbccdd]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256, rowsPerImage: 1), destination: array,
                                 region: .init(width: 1, height: 1, arrayLayer: 1, depth: 2)),
            .copyTextureToTexture(source: array, destination: volume,
                                  sourceRegion: .init(width: 1, height: 1, arrayLayer: 1, depth: 2),
                                  destinationOrigin: .init(z: 1)),
            .copyTextureToTexture(source: volume, destination: result,
                                  sourceRegion: .init(width: 1, height: 1, z: 1, depth: 2),
                                  destinationOrigin: .init(arrayLayer: 1)),
            .copyTextureToBuffer(source: result, destination: read, bytesPerRow: 256,
                                 arrayLayer: 1, depth: 2, rowsPerImage: 1),
            .read(buffer: read, byteCount: 260),
        ])
        let words = try await ticket.wait()[0]
        #expect(words[0] == 0x11223344)
        #expect(words[64] == 0xaabbccdd)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
