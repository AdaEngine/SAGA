import SAGATypes
import Testing

@Test func meshResolveAliasesRespectDestinationMip() throws {
    let slot = ComputeBindingSlot(binding: 0)
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm,
                                sampleCount: 4, meshStorageTextures: [
                                    .init(slot: slot, format: .rgba8Unorm, access: .writeOnly),
                                ])
    let target = ResourceID(device: 1, serial: 2)
    let destination = ResourceID(device: 1, serial: 3)
    let mip0 = ResourceID(device: 1, serial: 4)
    let mip1 = ResourceID(device: 1, serial: 5)
    let textures = [
        target: TextureDescriptor(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment, sampleCount: 4),
        destination: TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment, .storageBinding], mipLevelCount: 2),
    ]
    let views: [ResourceID: TextureViewResource] = [
        mip0: .init(texture: destination, levels: 0..<1),
        mip1: .init(texture: destination, levels: 1..<2),
    ]
    func validate(_ image: ResourceID) throws {
        try BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load,
                                   workgroups: .init(x: 1), meshTextures: [slot: image],
                                   resolve: .init(texture: destination, mipLevel: 1))
            .validate(device: 1, descriptors: [:], textures: textures, samplerDescriptors: [:], textureViews: views)
    }
    try validate(mip0)
    #expect(throws: GPUError.self) { try validate(mip1) }
}
