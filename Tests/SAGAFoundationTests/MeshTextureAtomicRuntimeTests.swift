import Testing
import SAGATypes
import SAGABackend

@Test func meshAtomicTexturesRequireAtomicUsageAndRejectAliasesAcrossStages() throws {
    let slot = ComputeBindingSlot(binding: 3)
    let atomic = StorageTextureRequirement(slot: slot, format: .r32Uint, access: .atomic)
    let target = ResourceID(device: 1, serial: 2)
    let image = ResourceID(device: 1, serial: 3)
    let color = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment)
    let atomicTexture = TextureDescriptor(width: 1, height: 1, format: .r32Uint, usage: .storageAtomic)
    let ordinaryTexture = TextureDescriptor(width: 1, height: 1, format: .r32Uint, usage: .storageBinding)
    for stage in 0..<3 {
        let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm,
            fragmentStorageTextures: stage == 2 ? [atomic] : [],
            taskStorageTextures: stage == 0 ? [atomic] : [], meshStorageTextures: stage == 1 ? [atomic] : [])
        let command = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1),
            taskTextures: stage == 0 ? [slot: image] : [:], meshTextures: stage == 1 ? [slot: image] : [:],
            fragmentTextures: stage == 2 ? [slot: image] : [:])
        try command.validate(device: 1, descriptors: [:], textures: [target: color, image: atomicTexture])
        #expect(throws: GPUError.self) { try command.validate(device: 1, descriptors: [:], textures: [target: color, image: ordinaryTexture]) }
    }
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm,
        taskStorageTextures: [atomic], meshStorageTextures: [atomic])
    let alias = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1),
        taskTextures: [slot: image], meshTextures: [slot: image])
    #expect(throws: GPUError.self) { try alias.validate(device: 1, descriptors: [:], textures: [target: color, image: atomicTexture]) }
}
