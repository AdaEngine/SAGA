import Testing
import SAGATypes
import SAGACore
import SAGAMetal

private let uploadTexels: [UInt32] = [0xff112233, 0xff445566, 0xff778899, 0xffaabbcc]

private func paddedTextureBytes() -> [UInt8] {
    var bytes = Array(repeating: UInt8(0xee), count: 25)
    for (index, offset) in [1, 6, 16, 21].enumerated() {
        let word = uploadTexels[index]
        for lane in 0..<4 { bytes[offset + lane] = UInt8(truncatingIfNeeded: word >> (lane * 8)) }
    }
    return bytes
}

@Test func writeTextureValidatesUnalignedRowsImagesAndDestinationExtents() throws {
    let texture = ResourceID(device: 1, serial: 1)
    let descriptor = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: .copyDestination,
                                       mipLevelCount: 2, arrayLayerCount: 4)
    let region = TextureCopyRegion(x: 1, width: 1, height: 2, mipLevel: 1, arrayLayer: 1, depth: 2)
    let bytes = paddedTextureBytes()
    let layout = TextureDataLayout(offset: 1, bytesPerRow: 5, rowsPerImage: 3)
    func write(_ layout: TextureDataLayout, region: TextureCopyRegion = region, data: [UInt8] = bytes) -> BufferCommand {
        .writeTexture(destination: texture, data: data, layout: layout, region: region)
    }
    let valid = write(layout)
    #expect(valid.resources == [texture])
    try valid.validate(device: 1, descriptors: [:], textures: [texture: descriptor])
    try write(.init(offset: 25), region: .init(x: 2, y: 2, width: 0, height: 0, mipLevel: 1, arrayLayer: 4, depth: 0))
        .validate(device: 1, descriptors: [:], textures: [texture: descriptor])
    for invalid in [
        write(.init(offset: 1, bytesPerRow: 5)),
        write(.init(offset: 1, rowsPerImage: 3)),
        write(.init(offset: 1, bytesPerRow: 3, rowsPerImage: 3)),
        write(.init(offset: 1, bytesPerRow: 5, rowsPerImage: 1)),
        write(.init(offset: .max, bytesPerRow: 5, rowsPerImage: 3)),
        write(.init(bytesPerRow: .max, rowsPerImage: .max),
              region: .init(width: 1, height: 2, mipLevel: 1, depth: 3)),
        write(layout, data: Array(bytes.dropLast())),
        write(layout, region: .init(x: 2, width: 1, height: 2, mipLevel: 1, depth: 2)),
        write(layout, region: .init(width: 1, height: 2, mipLevel: 1, arrayLayer: 3, depth: 2)),
        write(layout, region: .init(width: 1, height: 2, mipLevel: 1, z: 1, depth: 2)),
    ] {
        #expect(throws: GPUError.self) { try invalid.validate(device: 1, descriptors: [:], textures: [texture: descriptor]) }
    }
    #expect(throws: GPUError.self) {
        try valid.validate(device: 1, descriptors: [:], textures: [
            texture: .init(width: 4, height: 4, format: .rgba8Unorm, usage: .copySource, mipLevelCount: 2, arrayLayerCount: 4),
        ])
    }
    #expect(throws: GPUError.self) {
        try valid.validate(device: 1, descriptors: [:], textures: [
            texture: .init(width: 4, height: 4, format: .depth32Float, usage: .copyDestination, mipLevelCount: 2, arrayLayerCount: 4),
        ])
    }
    let stencil = TextureDescriptor(width: 3, height: 2, format: .depth24PlusStencil8, usage: .copyDestination)
    let stencilBytes: [UInt8] = [0xee, 1, 128, 240, 0xee, 0xee, 17, 7, 165]
    let stencilLayout = TextureDataLayout(offset: 1, bytesPerRow: 5)
    try BufferCommand.writeTexture(destination: texture, data: stencilBytes, layout: stencilLayout,
                                    region: .init(width: 3, height: 2), aspect: .stencilOnly)
        .validate(device: 1, descriptors: [:], textures: [texture: stencil])
    #expect(throws: GPUError.self) {
        try BufferCommand.writeTexture(destination: texture, data: stencilBytes, layout: stencilLayout,
                                        region: .init(width: 2, height: 2), aspect: .stencilOnly)
            .validate(device: 1, descriptors: [:], textures: [texture: stencil])
    }
}

#if canImport(Metal)
@Test func writesTextureMipsArrayLayersAndVolumeSlicesThroughTheQueueOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let array = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], mipLevelCount: 2, arrayLayerCount: 3))
    let volume = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], dimension: .d3, depth: 3))
    let stencil = try await device.createTexture(.init(width: 3, height: 2, format: .depth24PlusStencil8,
        usage: [.copySource, .copyDestination]))
    let readback = try await device.createBuffer(.init(size: 784, usage: [.mapRead, .copyDestination]))
    let bytes = paddedTextureBytes()
    let layout = TextureDataLayout(offset: 1, bytesPerRow: 5, rowsPerImage: 3)
    let volumeUpload = try await device.writeTexture(volume, data: bytes, layout: layout,
        region: .init(x: 1, width: 1, height: 2, z: 1, depth: 2))
    func capture(_ texture: ResourceID, mipLevel: UInt32 = 0, layer: UInt32 = 0, z: UInt32 = 0) -> [BufferCommand] {
        [
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256,
                                 mipLevel: mipLevel, arrayLayer: layer, z: z),
            .read(buffer: readback, byteCount: mipLevel == 0 && texture == array ? 784 : 264),
        ]
    }
    let upload = BufferCommand.writeTexture(destination: array, data: bytes, layout: layout,
        region: .init(x: 1, width: 1, height: 2, mipLevel: 1, arrayLayer: 1, depth: 2))
    var commands = capture(array, mipLevel: 1, layer: 1) + [upload]
    for layer: UInt32 in [0, 1, 2] { commands += capture(array, mipLevel: 1, layer: layer) }
    for plane: UInt32 in [0, 1, 2] { commands += capture(volume, z: plane) }
    commands += capture(array, layer: 1)
    commands += [
        .writeTexture(destination: stencil, data: [0xee, 1, 128, 240, 0xee, 0xee, 17, 7, 165],
                      layout: .init(offset: 1, bytesPerRow: 5), region: .init(width: 3, height: 2), aspect: .stencilOnly),
        .copyTextureToBuffer(source: stencil, destination: readback, bytesPerRow: 256, aspect: .stencilOnly),
        .read(buffer: readback, byteCount: 260),
    ]
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    #expect(volumeUpload.sequence < ticket.sequence)
    #expect(snapshots.count == 9)
    let lanes = snapshots.prefix(7).map { [$0[0], $0[1], $0[64], $0[65]] }
    #expect(lanes == [
        [0, 0, 0, 0], [0, 0, 0, 0],
        [0, uploadTexels[0], 0, uploadTexels[1]],
        [0, uploadTexels[2], 0, uploadTexels[3]],
        [0, 0, 0, 0],
        [0, uploadTexels[0], 0, uploadTexels[1]],
        [0, uploadTexels[2], 0, uploadTexels[3]],
    ])
    #expect(snapshots[7].allSatisfy { $0 == 0 })
    #expect([snapshots[8][0] & 0x00ffffff, snapshots[8][64] & 0x00ffffff] == [0x00f08001, 0x00a50711])
    await device.close()
}
#endif
