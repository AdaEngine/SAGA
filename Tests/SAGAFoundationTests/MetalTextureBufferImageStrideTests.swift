import Testing
import SAGATypes
import SAGACore
import SAGAMetal

@Test func validatesPaddedTextureBufferImageStrideInBlockRows() throws {
    let texture = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let descriptor = TextureDescriptor(width: 8, height: 8, format: .bc1RGBAUnorm,
        usage: [.copyDestination], arrayLayerCount: 2)
    func check(_ rows: UInt32, size: UInt64) throws {
        try BufferCommand.copyBufferToTexture(source: buffer,
            layout: .init(bytesPerRow: 256, rowsPerImage: rows), destination: texture,
            region: .init(width: 8, height: 8, depth: 2))
            .validate(device: 1, descriptors: [buffer: .init(size: size, usage: [.copySource])],
                      textures: [texture: descriptor])
    }
    try check(3, size: 1040)
    try BufferCommand.copyBufferToTexture(source: buffer,
        layout: .init(bytesPerRow: 256, rowsPerImage: .max), destination: texture,
        region: .init(width: 8, height: 8))
        .validate(device: 1, descriptors: [buffer: .init(size: 272, usage: [.copySource])],
                  textures: [texture: descriptor])
    #expect(throws: GPUError.self) { try check(3, size: 1039) }
    #expect(throws: GPUError.self) { try check(1, size: 2048) }
}

@Test func validatesPaddedTextureReadbackInBlockRows() throws {
    let texture = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let descriptor = TextureDescriptor(width: 8, height: 8, format: .bc1RGBAUnorm,
        usage: [.copySource], arrayLayerCount: 2)
    func check(_ rows: UInt32, size: UInt64) throws {
        try BufferCommand.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256,
            destinationOffset: 8, depth: 2, rowsPerImage: rows)
            .validate(device: 1, descriptors: [buffer: .init(size: size, usage: [.copyDestination])],
                      textures: [texture: descriptor])
    }
    try check(3, size: 1048)
    try BufferCommand.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256,
        destinationOffset: 8, rowsPerImage: .max)
        .validate(device: 1, descriptors: [buffer: .init(size: 280, usage: [.copyDestination])],
                  textures: [texture: descriptor])
    #expect(throws: GPUError.self) { try check(3, size: 1047) }
    #expect(throws: GPUError.self) { try check(1, size: 2048) }
}

#if canImport(Metal)
@Test(arguments: [TextureDimension.d2, .d3])
func copiesPaddedBufferImagesOnMetal(dimension: TextureDimension) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination], arrayLayerCount: dimension == .d2 ? 2 : 1,
            dimension: dimension, depth: dimension == .d3 ? 2 : 1))
        let upload = try await device.createBuffer(.init(size: 1032, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 1036, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [1, 2]),
            .write(buffer: upload, offset: 256, words: [3, 4]),
            .write(buffer: upload, offset: 768, words: [5, 6]),
            .write(buffer: upload, offset: 1024, words: [7, 8]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256, rowsPerImage: 3),
                destination: texture, region: .init(width: 2, height: 2, depth: 2)),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256,
                                 destinationOffset: 4, depth: 2, rowsPerImage: 3),
            .read(buffer: read, byteCount: 1036),
        ])
        let words = try await ticket.wait()[0]
        #expect([words[1], words[2], words[65], words[66], words[193], words[194], words[257], words[258]] ==
            [1, 2, 3, 4, 5, 6, 7, 8])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func copiesSingleImageWithMaximumRowsPerImageOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination]))
        let upload = try await device.createBuffer(.init(size: 4, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0x12345678]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 0xffffff00, rowsPerImage: .max),
                                 destination: texture, region: .init(width: 1, height: 1)),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 0xffffff00, rowsPerImage: .max),
            .read(buffer: read, byteCount: 4),
        ])
        #expect(try await ticket.wait()[0] == [0x12345678])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
