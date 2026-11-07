import Testing
import SAGATypes
import SAGACore
import SAGAMetal

@Test func validatesTextureUploadRegionAndLinearFootprint() throws {
    let source = ResourceID(device: 1, serial: 1)
    let target = ResourceID(device: 1, serial: 2)
    let region = TextureCopyRegion(x: 1, y: 1, width: 2, height: 2)
    let descriptors = [source: BufferDescriptor(size: 268, usage: .copySource)]
    let textures = [target: TextureDescriptor(width: 4, height: 3, format: .rgba8Unorm, usage: .copyDestination)]
    func copy(_ layout: TextureBufferLayout, _ area: TextureCopyRegion = region) -> BufferCommand {
        .copyBufferToTexture(source: source, layout: layout, destination: target, region: area)
    }
    let command = copy(.init(offset: 4, bytesPerRow: 256))
    #expect(command.resources == [source, target])
    try command.validate(device: 1, descriptors: descriptors, textures: textures)
    try copy(.init(offset: 268), .init(x: 4, y: 3, width: 0, height: 0))
        .validate(device: 1, descriptors: descriptors, textures: textures)
    for invalid in [
        copy(.init(offset: 2, bytesPerRow: 256)),
        copy(.init(offset: 4, bytesPerRow: 252)),
        copy(.init(offset: 4)),
        copy(.init(offset: 8, bytesPerRow: 256)),
        copy(.init(offset: 4, bytesPerRow: 256), .init(x: UInt32.max, width: 2, height: 2)),
        copy(.init(offset: 4, bytesPerRow: 256), .init(y: 2, width: 2, height: 2)),
        copy(.init(offset: UInt64.max - 3, bytesPerRow: 256)),
    ] {
        #expect(throws: GPUError.self) { try invalid.validate(device: 1, descriptors: descriptors, textures: textures) }
    }
    #expect(throws: GPUError.self) {
        try command.validate(device: 1, descriptors: [source: .init(size: 268, usage: .copyDestination)], textures: textures)
    }
    #expect(throws: GPUError.self) {
        try command.validate(device: 1, descriptors: descriptors, textures: [
            target: .init(width: 4, height: 3, format: .rgba8Unorm, usage: .copySource),
        ])
    }
}

#if canImport(Metal)
@Test(arguments: [TextureFormat.rgba8Unorm, .bgra8Unorm])
func uploadsPaddedTextureRowsAndPreservesUntouchedTexelsOnMetal(format: TextureFormat) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let source = try await device.createBuffer(.init(size: 268, usage: [.copySource, .copyDestination]))
        let target = try await device.createTexture(.init(width: 4, height: 3, format: format,
                                                          usage: [.copyDestination, .copySource]))
        let full = try await device.createTexture(.init(width: 2, height: 1, format: format,
                                                        usage: [.copyDestination, .copySource]))
        let read = try await device.createBuffer(.init(size: 528, usage: [.copyDestination, .mapRead]))
        let texels: [UInt32] = [0xff112233, 0xff445566, 0xff778899, 0xffaabbcc]
        let ticket = try await device.submitCommands([
            .write(buffer: source, offset: 4, words: Array(texels.prefix(2))),
            .write(buffer: source, offset: 260, words: Array(texels.suffix(2))),
            .copyBufferToTexture(source: source, layout: .init(offset: 4, bytesPerRow: 256),
                                 destination: target, region: .init(x: 1, y: 1, width: 2, height: 2)),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 528),
            .copyBufferToTexture(source: source, layout: .init(offset: 8),
                                 destination: target, region: .init(x: 3, width: 1, height: 1)),
            .copyBufferToTexture(source: source, layout: .init(offset: 268),
                                 destination: target, region: .init(x: 4, y: 3, width: 0, height: 0)),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 528),
            .copyBufferToTexture(source: source, layout: .init(offset: 4),
                                 destination: full, region: .init(width: 2, height: 1)),
            .copyTextureToBuffer(source: full, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
        ])
        try await device.destroyBuffer(source)
        let result = try await ticket.wait()
        var first = Array(repeating: UInt32(0), count: 132)
        first[65] = texels[0]; first[66] = texels[1]
        first[129] = texels[2]; first[130] = texels[3]
        var second = first
        second[3] = texels[1]
        #expect(result == [first, second, Array(texels.prefix(2))])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
