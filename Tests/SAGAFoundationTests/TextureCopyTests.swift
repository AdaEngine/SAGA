import Testing
import SAGATypes
import SAGACore
import SAGAMetal

@Test func validatesDisjointVolumeSliceCopies() throws {
    let image = ResourceID(device: 1, serial: 1)
    let volume = TextureDescriptor(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], dimension: .d3, depth: 4)
    func copy(_ z: UInt32, _ destinationZ: UInt32, depth: UInt32) throws {
        try BufferCommand.copyTextureToTexture(source: image, destination: image,
            sourceRegion: .init(width: 2, height: 1, z: z, depth: depth),
            destinationOrigin: .init(z: destinationZ))
            .validate(device: 1, descriptors: [:], textures: [image: volume])
    }
    try copy(0, 2, depth: 2)
    #expect(throws: GPUError.validation("Texture copy subresources must be disjoint")) {
        try copy(0, 1, depth: 2)
    }
    #expect(throws: GPUError.validation("Texture copy subresources must be disjoint")) {
        try copy(0, 0, depth: 0)
    }
}

@Test func validatesTextureCopySubresourcesFormatsAndDepthExtent() throws {
    let source = ResourceID(device: 1, serial: 1)
    let destination = ResourceID(device: 1, serial: 2)
    let usage: TextureUsage = [.copySource, .copyDestination]
    let color = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: usage, mipLevelCount: 2)
    let depth = TextureDescriptor(width: 4, height: 4, format: .depth32Float, usage: usage, mipLevelCount: 2)
    func validate(_ command: BufferCommand, _ descriptors: [ResourceID: TextureDescriptor]) throws {
        try command.validate(device: 1, descriptors: [:], textures: descriptors)
    }
    try validate(.copyTextureToTexture(source: source, destination: source,
        sourceRegion: .init(x: 1, y: 1, width: 2, height: 2), destinationOrigin: .init(mipLevel: 1)), [source: color])
    #expect(throws: GPUError.validation("Texture copy subresources must be disjoint")) {
        try validate(.copyTextureToTexture(source: source, destination: source,
            sourceRegion: .init(width: 1, height: 1), destinationOrigin: .init(x: 3, y: 3)), [source: color])
    }
    let full = BufferCommand.copyTextureToTexture(source: source, destination: destination,
                                                  sourceRegion: .init(width: 4, height: 4))
    #expect(throws: GPUError.self) { try validate(full, [source: color, destination: depth]) }
    try validate(full, [source: depth, destination: depth])
    let multisampled = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm,
        usage: [.copySource, .renderAttachment], sampleCount: 4)
    #expect(throws: GPUError.self) { try validate(full, [source: multisampled, destination: color]) }
    #expect(throws: GPUError.validation("Depth and multisampled texture copies require full mip extents")) {
        try validate(.copyTextureToTexture(source: source, destination: destination,
            sourceRegion: .init(width: 2, height: 2)), [source: depth, destination: depth])
    }
}

#if canImport(Metal)
@Test func copiesDisjointVolumeSlicesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination], dimension: .d3, depth: 2))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .writeTexture(destination: image, data: [1, 2, 3, 4], layout: .init(),
                          region: .init(width: 1, height: 1)),
            .copyTextureToTexture(source: image, destination: image,
                sourceRegion: .init(width: 1, height: 1), destinationOrigin: .init(z: 1)),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, z: 1),
            .read(buffer: read, byteCount: 4),
        ])
        #expect(try await ticket.wait() == [[0x04030201]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func copiesColorRegionsBetweenMipsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination], mipLevelCount: 2))
        let destination = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination]))
        let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff00ff00, 0xff0000ff, 0xffff0000, 0xffffffff]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(y: 1, width: 4, height: 1)),
            .copyTextureToTexture(source: image, destination: image,
                                  sourceRegion: .init(x: 1, y: 1, width: 1, height: 1),
                                  destinationOrigin: .init(x: 1, mipLevel: 1)),
            .copyTextureToTexture(source: image, destination: destination,
                                  sourceRegion: .init(width: 2, height: 1, mipLevel: 1)),
            .copyTextureToBuffer(source: destination, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
        ])
        #expect(try await ticket.wait() == [[0, 0xff0000ff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
