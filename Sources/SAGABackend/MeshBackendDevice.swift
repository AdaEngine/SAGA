import SAGATypes
import SAGAShaderIR

public protocol MeshBackendDevice: CommandBackendDevice {
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact,
                              fragment: RenderStageArtifact) async throws
    func destroyMeshPipeline(id: ResourceID) async
}

package func meshBufferRequirements(
    uniforms: [ComputeShaderArtifact.UniformBinding], storage: [ComputeShaderArtifact.StorageBinding]
) throws -> [ComputePipeline.BufferBinding] {
    var slots = Set<ComputeBindingSlot>()
    var result: [ComputePipeline.BufferBinding] = []
    for binding in uniforms {
        let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
        guard binding.flatBindingIndex != nil, slots.insert(slot).inserted else {
            throw GPUError.unsupported("Mesh stage uniform binding layout")
        }
        result += try binding.binding.computeSlots(array: binding.bindingArray, limit: ComputeShaderArtifact.uniformBindingCapacity).map {
            .init(slot: $0, minimumByteSize: binding.minimumByteSize, isReadOnly: true, usage: .uniform)
        }
    }
    for binding in storage {
        let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
        guard binding.flatBindingIndex != nil,
              binding.memoryDecorations.isEmpty, slots.insert(slot).inserted else {
            throw GPUError.unsupported("Mesh stage storage binding layout")
        }
        var minimum = binding.minimumByteSize
        if let stride = binding.runtimeArrayStride {
            guard stride > 0 else { throw GPUError.validation("Invalid mesh runtime array stride") }
            minimum = max(minimum, UInt64(binding.runtimeArrayOffset) + UInt64(stride))
        }
        result += try binding.binding.computeSlots(array: binding.bindingArray, limit: ComputeShaderArtifact.storageBindingCapacity).map {
            .init(slot: $0, minimumByteSize: minimum, isReadOnly: binding.isReadOnly)
        }
    }
    return result
}

package func meshHandleRequirements(
    textures: [ComputeShaderArtifact.SampledTextureBinding], samplers: [ComputeShaderArtifact.SamplerBinding]
) throws -> ([SampledTextureRequirement], [SamplerRequirement]) {
    let samplerRequirements = try samplers.flatMap { try $0.requirements() }
    let nonFiltering = Set(samplerRequirements.filter(\.requiresNonFiltering).map(\.slot))
    let whollyNonFiltering = whollyNonFilteringSamplerBindings(nonFiltering, in: samplerRequirements)
    return (try textures.flatMap { try $0.requirements(nonFilteringSamplers: whollyNonFiltering) }, samplerRequirements)
}

package func meshStorageTextureRequirements(
    _ bindings: [ComputeShaderArtifact.StorageTextureBinding]
) throws -> [StorageTextureRequirement] {
    try bindings.flatMap { try $0.requirements() }
}
