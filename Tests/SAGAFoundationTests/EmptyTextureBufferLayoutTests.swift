import Testing
import SAGATypes

@Test func emptyHeightStillValidatesInterImageBufferFootprint() throws {
    let texture = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let command = BufferCommand.copyBufferToTexture(
        source: buffer, layout: .init(bytesPerRow: 256, rowsPerImage: 3), destination: texture,
        region: .init(width: 1, height: 0, depth: 2))
    let textures = [texture: TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm,
                                               usage: [.copyDestination], arrayLayerCount: 2)]
    #expect(throws: GPUError.self) {
        try command.validate(device: 1, descriptors: [buffer: .init(size: 767, usage: [.copySource])],
                             textures: textures)
    }
    try command.validate(device: 1, descriptors: [buffer: .init(size: 768, usage: [.copySource])],
                         textures: textures)
}

@Test func emptyRegionReadbackStillValidatesInterImageBufferFootprint() throws {
    let texture = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let command = BufferCommand.copyTextureRegionToBuffer(
        source: texture, destination: buffer, layout: .init(bytesPerRow: 256, rowsPerImage: 3),
        region: .init(width: 1, height: 0, depth: 2))
    let textures = [texture: TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm,
                                                usage: .copySource, arrayLayerCount: 2)]
    #expect(throws: GPUError.self) {
        try command.validate(device: 1, descriptors: [buffer: .init(size: 767, usage: .copyDestination)],
                             textures: textures)
    }
    try command.validate(device: 1, descriptors: [buffer: .init(size: 768, usage: .copyDestination)],
                         textures: textures)
}

@Test func emptyDepthReadbackRequiresFullWidthAndHeight() throws {
    let texture = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let textures = [texture: TextureDescriptor(width: 2, height: 2, format: .depth32Float, usage: .copySource)]
    let buffers = [buffer: BufferDescriptor(size: 512, usage: .copyDestination)]
    func check(_ region: TextureCopyRegion) throws {
        try BufferCommand.copyTextureRegionToBuffer(source: texture, destination: buffer,
            layout: .init(bytesPerRow: 256), region: region, aspect: .depthOnly)
            .validate(device: 1, descriptors: buffers, textures: textures)
    }
    try check(.init(width: 2, height: 2, depth: 0))
    #expect(throws: GPUError.self) { try check(.init(width: 2, height: 0)) }
    #expect(throws: GPUError.self) { try check(.init(width: 0, height: 2)) }
}
