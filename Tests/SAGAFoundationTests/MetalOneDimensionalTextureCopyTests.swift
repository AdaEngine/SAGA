import Testing
import SAGATypes
import SAGACore
import SAGAMetal

#if canImport(Metal)
@Test func oneDimensionalTextureCopiesAcrossDimensionsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let usage: TextureUsage = [.copySource, .copyDestination]
        let line = try await device.createTexture(.init(width: 4, height: 1, format: .rgba8Unorm,
            usage: usage, dimension: .d1))
        let fromArray = try await device.createTexture(.init(width: 4, height: 1, format: .rgba8Unorm,
            usage: usage, dimension: .d1))
        let fromVolume = try await device.createTexture(.init(width: 4, height: 1, format: .rgba8Unorm,
            usage: usage, dimension: .d1))
        let array = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: usage, arrayLayerCount: 2))
        let volume = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: usage, dimension: .d3, depth: 2))
        let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0x01020304, 0x11223344, 0xaabbccdd, 0x55667788]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: line,
                                 region: .init(width: 4, height: 1)),
            .copyTextureToTexture(source: line, destination: array,
                sourceRegion: .init(x: 1, width: 2, height: 1), destinationOrigin: .init(y: 1, arrayLayer: 1)),
            .copyTextureToTexture(source: array, destination: fromArray,
                sourceRegion: .init(y: 1, width: 2, height: 1, arrayLayer: 1), destinationOrigin: .init(x: 1)),
            .copyTextureToTexture(source: line, destination: volume,
                sourceRegion: .init(x: 1, width: 2, height: 1), destinationOrigin: .init(y: 1, z: 1)),
            .copyTextureToTexture(source: volume, destination: fromVolume,
                sourceRegion: .init(y: 1, width: 2, height: 1, z: 1), destinationOrigin: .init(x: 1)),
            .copyTextureToBuffer(source: fromArray, destination: read, bytesPerRow: 256),
            .copyTextureToBuffer(source: fromVolume, destination: read, bytesPerRow: 256, destinationOffset: 16),
            .read(buffer: read, byteCount: 32),
        ])
        let words = try await ticket.wait()[0]
        #expect(words == [0, 0x11223344, 0xaabbccdd, 0, 0, 0x11223344, 0xaabbccdd, 0])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
