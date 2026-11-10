import Testing
import SAGATypes
import SAGACore
import SAGAMetal

@Test func emptyTextureReadbackValidatesWithoutCopying() throws {
    let texture = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let textures = [texture: TextureDescriptor(width: 4, height: 4, format: .bc1RGBAUnorm,
        usage: [.copySource], arrayLayerCount: 2)]
    let buffers = [buffer: BufferDescriptor(size: 8, usage: [.copyDestination])]
    func validate(_ offset: UInt64, layer: UInt32, stride: UInt32 = 256) throws {
        try BufferCommand.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: stride,
            destinationOffset: offset, arrayLayer: layer, depth: 0)
            .validate(device: 1, descriptors: buffers, textures: textures)
    }
    try validate(8, layer: 2)
    #expect(throws: GPUError.self) { try validate(16, layer: 2) }
    #expect(throws: GPUError.self) { try validate(8, layer: 3) }
    #expect(throws: GPUError.self) { try validate(8, layer: 2, stride: 12) }
}

#if canImport(Metal)
@Test(arguments: [TextureDimension.d2, .d3])
func emptyTextureReadbackDoesNotTouchMetalBuffer(dimension: TextureDimension) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.copySource], dimension: dimension))
        let buffer = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: buffer, offset: 0, words: [0x12345678]),
            .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256,
                                 destinationOffset: 4, depth: 0),
            .read(buffer: buffer, byteCount: 4),
        ])
        #expect(try await ticket.wait()[0] == [0x12345678])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
