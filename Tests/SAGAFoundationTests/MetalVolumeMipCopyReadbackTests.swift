import Testing
import SAGATypes
import SAGACore
import SAGAMetal

#if canImport(Metal)
@Test func copiesSingleVolumeMipPlaneIntoExactlySizedBufferOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let texture = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination], mipLevelCount: 2, dimension: .d3, depth: 4))
        let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0x11223344, 0x55667788]),
            .write(buffer: upload, offset: 256, words: [0x99aabbcc, 0xddeeff00]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: texture,
                                 region: .init(width: 2, height: 2, mipLevel: 1, z: 1)),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1, z: 1),
            .read(buffer: read, byteCount: 264),
        ])
        let words = try await ticket.wait()[0]
        #expect([words[0], words[1], words[64], words[65]] ==
                [0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
