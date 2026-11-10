import Testing
import SAGATypes
import SAGABackend

@Test(arguments: [TextureViewDimension.d1, .d2Array, .d3])
func meshStorageTextureShapesValidateViews(dimension: TextureViewDimension) throws {
    let slot = ComputeBindingSlot(binding: 3)
    let requirement = StorageTextureRequirement(slot: slot, format: .rgba8Unorm, access: .writeOnly, dimension: dimension)
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm,
                                meshStorageTextures: [requirement])
    let target = ResourceID(device: 1, serial: 2), image = ResourceID(device: 1, serial: 3)
    let view = ResourceID(device: 1, serial: 4)
    let textures = [target: TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment),
                    image: TextureDescriptor(width: 2, height: 1, format: .rgba8Unorm, usage: .storageBinding)]
    let command = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load,
                                         workgroups: .init(x: 1), meshTextures: [slot: view])
    func validate(_ viewDimension: TextureViewDimension, levels: Range<UInt32> = 0..<1) throws {
        try command.validate(device: 1, descriptors: [:], textures: textures, samplerDescriptors: [:],
                             textureViews: [view: .init(texture: image, levels: levels, dimension: viewDimension)])
    }
    #expect(command.resources.contains(view))
    try validate(dimension)
    #expect(throws: GPUError.self) { try validate(.d2) }
    #expect(throws: GPUError.self) { try validate(dimension, levels: 0..<2) }
}
