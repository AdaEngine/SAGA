import Testing
import SAGATypes
import SAGACore
import SAGAMetal

@Test func validatesBufferCopiesAcrossTextureArrayLayers() throws {
    let texture = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let descriptor = TextureDescriptor(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], arrayLayerCount: 4)
    let buffers = [buffer: BufferDescriptor(size: 780, usage: [.copySource, .copyDestination])]
    func check(_ command: BufferCommand) throws {
        try command.validate(device: 1, descriptors: buffers, textures: [texture: descriptor])
    }
    try check(.copyBufferToTexture(source: buffer, layout: .init(offset: 4, bytesPerRow: 256),
        destination: texture, region: .init(width: 2, height: 2, arrayLayer: 1, depth: 2)))
    try check(.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256,
        destinationOffset: 4, arrayLayer: 1, depth: 2))
    #expect(throws: GPUError.self) {
        try check(.copyBufferToTexture(source: buffer, layout: .init(bytesPerRow: 256),
            destination: texture, region: .init(width: 2, height: 2, arrayLayer: 3, depth: 2)))
    }
    #expect(throws: GPUError.self) {
        try check(.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256,
            destinationOffset: 8, arrayLayer: 1, depth: 2))
    }
}

#if canImport(Metal)
@Test func copiesBufferAcrossTextureArrayLayersOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination], arrayLayerCount: 4))
        let upload = try await device.createBuffer(.init(size: 780, usage: [.copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 780, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 4, words: [1, 2]),
            .write(buffer: upload, offset: 260, words: [3, 4]),
            .write(buffer: upload, offset: 516, words: [5, 6]),
            .write(buffer: upload, offset: 772, words: [7, 8]),
            .copyBufferToTexture(source: upload, layout: .init(offset: 4, bytesPerRow: 256),
                destination: texture, region: .init(width: 2, height: 2, arrayLayer: 1, depth: 2)),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256,
                destinationOffset: 4, arrayLayer: 1, depth: 2),
            .read(buffer: readback, byteCount: 780),
        ])
        let result = try await ticket.wait()[0]
        #expect([result[1], result[2], result[65], result[66], result[129], result[130], result[193], result[194]] ==
            [1, 2, 3, 4, 5, 6, 7, 8])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
