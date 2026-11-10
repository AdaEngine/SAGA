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

/// Merges the per-stage bindings of a mesh pipeline into the graphics pipeline model that ordinary draw paths understand.
package func meshRenderSurrogate(_ pipeline: MeshPipeline, colorFormat: TextureFormat) -> RenderPipeline {
        // Merge the per-stage binding lists into the graphics pipeline model the ordinary draw path understands.
        var uniforms: [ComputeBindingSlot: RenderPipeline.Uniform] = [:]
        var storage: [ComputeBindingSlot: RenderPipeline.StorageBufferBinding] = [:]
        func add(_ bindings: [ComputePipeline.BufferBinding], visibility: UInt32) {
            for binding in bindings {
                let handle = RenderPipeline.HandleBinding(slot: binding.slot, vertexIndex: nil, fragmentIndex: nil, visibility: visibility)
                if binding.usage == .uniform {
                    let previous = uniforms[binding.slot]
                    uniforms[binding.slot] = .init(slot: binding.slot, minimumByteSize: max(previous?.minimumByteSize ?? 0, binding.minimumByteSize),
                                                   vertexIndex: nil, fragmentIndex: nil, visibility: (previous?.visibility ?? 0) | visibility)
                } else {
                    storage[binding.slot] = .init(handle: handle, minimumByteSize: max(storage[binding.slot]?.minimumByteSize ?? 0, binding.minimumByteSize),
                                                  isReadOnly: binding.isReadOnly && (storage[binding.slot]?.isReadOnly ?? true))
                }
            }
        }
        add(pipeline.taskBufferBindings, visibility: 1)
        add(pipeline.meshBufferBindings, visibility: 1)
        add(pipeline.fragmentBufferBindings, visibility: 2)
        var textures: [ComputeBindingSlot: RenderPipeline.SampledTextureBinding] = [:]
        for requirement in pipeline.taskTextures + pipeline.meshTextures + pipeline.fragmentTextures {
            textures[requirement.slot] = .init(handle: .init(slot: requirement.slot, vertexIndex: nil, fragmentIndex: nil, visibility: 3),
                                               requirement: requirement)
        }
        var samplers: [ComputeBindingSlot: RenderPipeline.SamplerBinding] = [:]
        for requirement in pipeline.taskSamplers + pipeline.meshSamplers + pipeline.fragmentSamplers {
            samplers[requirement.slot] = .init(handle: .init(slot: requirement.slot, vertexIndex: nil, fragmentIndex: nil, visibility: 3),
                                               requirement: requirement)
        }
        func ordered<T>(_ values: [ComputeBindingSlot: T]) -> [T] {
            values.sorted { ($0.key.group, $0.key.binding, $0.key.arrayElement ?? 0) < ($1.key.group, $1.key.binding, $1.key.arrayElement ?? 0) }
                .map(\.value)
        }
        return RenderPipeline(
            id: pipeline.id, colorFormat: colorFormat, depthState: pipeline.depthState, sampleCount: pipeline.sampleCount,
            sampleMask: pipeline.sampleMask, alphaToCoverageEnabled: pipeline.alphaToCoverageEnabled,
            primitive: pipeline.primitive, uniforms: ordered(uniforms), storageBuffers: ordered(storage),
            sampledTextures: ordered(textures), samplerBindings: ordered(samplers))
}
