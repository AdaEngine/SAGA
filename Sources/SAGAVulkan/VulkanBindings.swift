import SAGATypes
import SAGABackend
import SAGAShaderIR

// Samplers and descriptor planning for buffers, sampled textures and storage textures.

struct VkSamplerCreateInfo {
    var sType: UInt32 = 31
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var magFilter: UInt32 = 0
    var minFilter: UInt32 = 0
    var mipmapMode: UInt32 = 0
    var addressModeU: UInt32 = 0
    var addressModeV: UInt32 = 0
    var addressModeW: UInt32 = 0
    var mipLodBias: Float = 0
    var anisotropyEnable: UInt32 = 0
    var maxAnisotropy: Float = 1
    var compareEnable: UInt32 = 0
    var compareOp: UInt32 = 0
    var minLod: Float = 0
    var maxLod: Float = 0
    var borderColor: UInt32 = 0
    var unnormalizedCoordinates: UInt32 = 0
}

struct VkDescriptorImageInfo {
    var sampler: UInt64 = 0
    var imageView: UInt64 = 0
    var imageLayout: UInt32 = VK.Image.layoutGeneral
}

struct DescriptorTotals {
    var sets: UInt32 = 0
    var counts: [UInt32: UInt32] = [:]
    mutating func add(_ type: UInt32, _ count: UInt32 = 1) { counts[type, default: 0] += count }
}

extension VulkanDevice: SamplerBackendDevice {
    struct SamplerState {
        var descriptor: SamplerDescriptor
        var handle: UInt64
    }

    public func allocateSampler(id: ResourceID, descriptor: SamplerDescriptor) async throws {
        try requireActive()
        try descriptor.validate()
        guard samplers[id] == nil else { throw GPUError.validation("Duplicate Vulkan sampler") }
        var info = VkSamplerCreateInfo()
        info.magFilter = descriptor.magFilter.rawValue
        info.minFilter = descriptor.minFilter.rawValue
        info.mipmapMode = descriptor.mipmapFilter.rawValue
        func address(_ mode: SamplerAddressMode) -> UInt32 {
            switch mode { case .repeat: 0; case .mirrorRepeat: 1; case .clampToEdge: 2; case .clampToBorder: 3 }
        }
        info.addressModeU = address(descriptor.addressModeU)
        info.addressModeV = address(descriptor.addressModeV)
        info.addressModeW = address(descriptor.addressModeW)
        // The float border colors: transparent black, opaque black and opaque white (VkBorderColor 0, 2 and 4).
        switch descriptor.borderColor {
        case .transparentBlack?: info.borderColor = 0
        case .opaqueBlack?: info.borderColor = 2
        case .opaqueWhite?: info.borderColor = 4
        case nil: break
        }
        info.minLod = descriptor.lodMinClamp
        info.maxLod = descriptor.lodMaxClamp
        // Anisotropy needs an optional device feature this backend does not enable, so it clamps to 1.
        if let compare = descriptor.compare {
            info.compareEnable = 1
            info.compareOp = compare.rawValue
        }
        var handle: UInt64 = 0
        let g = native.g, device = native.device
        try vulkanCheck(withUnsafePointer(to: &info) { g.createSampler(device, UnsafeRawPointer($0), nil, &handle) },
                        "sampler creation")
        samplers[id] = SamplerState(descriptor: descriptor, handle: handle)
    }

    public func destroySampler(id: ResourceID) async { destroySamplerNow(id: id) }

    func destroySamplerNow(id: ResourceID) {
        guard let sampler = samplers.removeValue(forKey: id) else { return }
        native.g.destroySampler(native.device, sampler.handle, nil)
    }
}

extension VulkanDevice {
    /// Resolves every buffer, texture and sampler binding of one draw or dispatch into descriptor writes.
    func planBindings(
        types: [ComputeBindingSlot: UInt32], buffers bufferBindings: [(slot: ComputeBindingSlot, usage: BufferUsage)],
        suppliedBuffers: [ComputeBindingSlot: StorageBufferBinding], sampledTextures: [ComputeBindingSlot],
        storageTextures: [ComputeBindingSlot], samplerSlots: [ComputeBindingSlot],
        textureResources: [ComputeBindingSlot: ResourceID], samplerResources: [ComputeBindingSlot: ResourceID],
        externalSlots: [ComputeBindingSlot] = [], accelerationSlots: [ComputeBindingSlot] = [],
        accelerationResources: [ComputeBindingSlot: ResourceID] = [:], totals: inout DescriptorTotals
    ) throws -> [DescriptorWrite] {
        var writes: [DescriptorWrite] = []
        for (slot, usage) in bufferBindings {
            guard let supplied = suppliedBuffers[slot], let buffer = buffers[supplied.buffer], let type = types[slot] else {
                throw GPUError.validation("Missing Vulkan buffer binding")
            }
            let range = try supplied.range(in: buffer.descriptor, usage: usage)
            let limit = usage == .uniform ? native.limits.maxUniformBufferRange : native.limits.maxStorageBufferRange
            guard range.count <= UInt64(limit) else { throw GPUError.validation("Buffer binding exceeds the Vulkan device limit") }
            totals.add(type)
            writes.append(.init(set: Int(slot.group), binding: slot.binding, type: type, buffer: buffer.native.handle,
                                offset: range.lowerBound, range: UInt64(range.count),
                                arrayElement: slot.arrayElement ?? 0))
        }
        func imageWrites(_ slots: [ComputeBindingSlot], usage: TextureUsage) throws {
            for slot in slots {
                guard let id = textureResources[slot], let texture = self.texture(for: id), texture.descriptor.usage.contains(usage),
                      let type = types[slot] else {
                    throw GPUError.validation("Missing or invalid Vulkan texture binding")
                }
                totals.add(type)
                writes.append(.init(set: Int(slot.group), binding: slot.binding, type: type, buffer: 0, offset: 0, range: 0,
                                    image: texture.image, imageFormat: texture.format.vk, imageAspect: imageAspect(texture), baseMip: texture.baseMip,
                                    viewType: viewType(texture), baseLayer: texture.baseLayer,
                                    layerCount: texture.viewLayers ?? (texture.descriptor.dimension == .d3 ? 1 : texture.descriptor.arrayLayerCount),
                                    mipLevels: texture.viewMips ?? texture.descriptor.mipLevelCount,
                                    arrayElement: slot.arrayElement ?? 0))
            }
        }
        try imageWrites(sampledTextures, usage: .textureBinding)
        try imageWrites(storageTextures, usage: .storageBinding)
        for slot in samplerSlots {
            guard let id = samplerResources[slot], let sampler = samplers[id], let type = types[slot] else {
                throw GPUError.validation("Missing Vulkan sampler binding")
            }
            totals.add(type)
            writes.append(.init(set: Int(slot.group), binding: slot.binding, type: type, buffer: 0, offset: 0, range: 0,
                                sampler: sampler.handle, arrayElement: slot.arrayElement ?? 0))
        }
        for slot in accelerationSlots {
            guard let id = accelerationResources[slot], let structure = accelerationStructures[id] else {
                throw GPUError.validation("Missing Vulkan acceleration structure binding")
            }
            totals.add(Self.accelerationDescriptorType)
            writes.append(.init(set: Int(slot.group), binding: slot.binding, type: Self.accelerationDescriptorType, buffer: 0, offset: 0,
                                range: 0, accelerationStructure: structure.handle))
        }
        // An external texture is three plane images and a block of parameter words at derived binding numbers.
        for slot in externalSlots {
            guard let id = textureResources[slot], let external = externalTextures[id] else {
                throw GPUError.validation("Missing Vulkan external texture binding")
            }
            let bindings = ComputeShaderArtifact.ExternalTextureLayout.vulkanBindings(for: IRResourceBinding(group: slot.group, binding: slot.binding))
            for index in 0..<3 {
                let planeID = external.planes[index < external.planes.count ? index : 0]
                guard let texture = self.texture(for: planeID), texture.descriptor.usage.contains(.textureBinding) else {
                    throw GPUError.validation("Invalid Vulkan external texture plane")
                }
                totals.add(VK.DescriptorType.sampledImage)
                writes.append(.init(set: Int(slot.group), binding: bindings.planes[index], type: VK.DescriptorType.sampledImage, buffer: 0,
                                    offset: 0, range: 0, image: texture.image, imageFormat: texture.format.vk,
                                    imageAspect: imageAspect(texture), baseMip: texture.baseMip, viewType: viewType(texture),
                                    baseLayer: texture.baseLayer, layerCount: texture.viewLayers ?? texture.descriptor.arrayLayerCount,
                                    mipLevels: texture.viewMips ?? texture.descriptor.mipLevelCount))
            }
            totals.add(VK.DescriptorType.uniformBuffer)
            writes.append(.init(set: Int(slot.group), binding: bindings.parameters, type: VK.DescriptorType.uniformBuffer,
                                buffer: external.parameters.handle, offset: 0,
                                range: UInt64(ComputeShaderArtifact.ExternalTextureLayout.parameterByteSize)))
        }
        return writes
    }

    /// `planBindings` for a pipeline that binds only buffers, reading the supplied bindings straight from a command list.
    func planBufferBindings(
        _ pipeline: Pipeline, supplied: ArraySlice<CommandList.Binding>, totals: inout DescriptorTotals
    ) throws -> [DescriptorWrite] {
        var writes: [DescriptorWrite] = []
        writes.reserveCapacity(pipeline.bufferBindings.count)
        for (slot, usage) in pipeline.bufferBindings {
            guard let binding = supplied.first(where: { $0.slot == slot }), let buffer = buffers[binding.buffer],
                  let type = pipeline.descriptorTypes[slot] else {
                throw GPUError.validation("Missing Vulkan buffer binding")
            }
            let range = try StorageBufferBinding(buffer: binding.buffer, offset: binding.offset, size: binding.size)
                .range(in: buffer.descriptor, usage: usage)
            let limit = usage == .uniform ? native.limits.maxUniformBufferRange : native.limits.maxStorageBufferRange
            guard range.count <= UInt64(limit) else { throw GPUError.validation("Buffer binding exceeds the Vulkan device limit") }
            totals.add(type)
            writes.append(.init(set: Int(slot.group), binding: slot.binding, type: type, buffer: buffer.native.handle,
                                offset: range.lowerBound, range: UInt64(range.count), arrayElement: slot.arrayElement ?? 0))
        }
        return writes
    }

    /// Allocates one descriptor set per layout from the batch pool, fills them and binds them at set zero.
    /// Image views are created per use and released with the batch.
    func bindDescriptors(commandBuffer: UInt, bindPoint: UInt32, layout: UInt64, setLayouts: [UInt64],
                         writes writeList: [DescriptorWrite], batch: inout PendingBatch) throws {
        guard !setLayouts.isEmpty else { return }
        let f = native.f, g = native.g, device = native.device
        var sets = [UInt64](repeating: 0, count: setLayouts.count)
        let pool = batch.descriptorPool
        try setLayouts.withUnsafeBufferPointer { layouts in
            var info = VkDescriptorSetAllocateInfo()
            info.descriptorPool = pool
            info.descriptorSetCount = UInt32(layouts.count)
            info.pSetLayouts = layouts.baseAddress
            try vulkanCheck(withUnsafePointer(to: &info) { f.allocateDescriptorSets(device, UnsafeRawPointer($0), &sets) },
                            "descriptor set allocation")
        }
        var accelerationHandles = writeList.map(\.accelerationStructure)
        var accelerationInfos = [VkWriteDescriptorSetAccelerationStructure](repeating: .init(), count: writeList.count)
        var bufferInfos = [VkDescriptorBufferInfo](repeating: .init(), count: writeList.count)
        var imageInfos = [VkDescriptorImageInfo](repeating: .init(), count: writeList.count)
        for (index, write) in writeList.enumerated() {
            switch write.type {
            case VK.DescriptorType.sampler:
                imageInfos[index].sampler = write.sampler
            case VK.DescriptorType.sampledImage, VK.DescriptorType.storageImage:
                var info = VkImageViewCreateInfo()
                info.image = write.image
                info.format = write.imageFormat
                info.viewType = write.viewType
                info.subresourceRange = VkImageSubresourceRange(aspectMask: write.imageAspect, baseMipLevel: write.baseMip,
                    levelCount: write.type == VK.DescriptorType.sampledImage ? write.mipLevels : 1,
                    baseArrayLayer: write.baseLayer, layerCount: write.layerCount)
                var view: UInt64 = 0
                try vulkanCheck(withUnsafePointer(to: &info) { g.createImageView(device, UnsafeRawPointer($0), nil, &view) },
                                "image view creation")
                batch.views.append(view)
                imageInfos[index].imageView = view
            default:
                bufferInfos[index] = VkDescriptorBufferInfo(buffer: write.buffer, offset: write.offset, range: write.range)
            }
        }
        accelerationHandles.withUnsafeMutableBufferPointer { handlePointer in
            for index in accelerationInfos.indices {
                accelerationInfos[index].pAccelerationStructures = UnsafeRawPointer(handlePointer.baseAddress! + index)
            }
        bufferInfos.withUnsafeBufferPointer { bufferPointer in
            imageInfos.withUnsafeBufferPointer { imagePointer in
              accelerationInfos.withUnsafeBufferPointer { accelerationPointer in
                let writes = writeList.enumerated().map { index, write in
                    var native = VkWriteDescriptorSet()
                    native.dstSet = sets[write.set]
                    native.dstBinding = write.binding
                    native.dstArrayElement = write.arrayElement
                    native.descriptorType = write.type
                    switch write.type {
                    case VK.DescriptorType.sampler, VK.DescriptorType.sampledImage, VK.DescriptorType.storageImage:
                        native.pImageInfo = UnsafeRawPointer(imagePointer.baseAddress! + index)
                    case Self.accelerationDescriptorType:
                        native.pNext = UnsafeRawPointer(accelerationPointer.baseAddress! + index)
                    default:
                        native.pBufferInfo = bufferPointer.baseAddress! + index
                    }
                    return native
                }
                writes.withUnsafeBufferPointer {
                    f.updateDescriptorSets(device, UInt32($0.count), UnsafeRawPointer($0.baseAddress), 0, nil)
                }
              }
            }
        }
        }
        sets.withUnsafeBufferPointer {
            f.cmdBindDescriptorSets(commandBuffer, bindPoint, layout, 0, UInt32($0.count), $0.baseAddress!, 0, nil)
        }
    }
}

/// Descriptor types of the plane images and parameter block of each external texture slot.
func vulkanExternalDescriptorTypes(_ slots: [ComputeBindingSlot]) -> [ComputeBindingSlot: UInt32] {
    var types: [ComputeBindingSlot: UInt32] = [:]
    for slot in slots {
        let bindings = ComputeShaderArtifact.ExternalTextureLayout.vulkanBindings(for: IRResourceBinding(group: slot.group, binding: slot.binding))
        for binding in bindings.planes { types[ComputeBindingSlot(group: slot.group, binding: binding)] = VK.DescriptorType.sampledImage }
        types[ComputeBindingSlot(group: slot.group, binding: bindings.parameters)] = VK.DescriptorType.uniformBuffer
    }
    return types
}

/// Descriptor types for the texture and sampler bindings of a pipeline, or an error for layouts Vulkan cannot express yet.
func vulkanHandleDescriptorTypes(
    sampled: [SampledTextureRequirement], storage: [StorageTextureRequirement], samplers: [SamplerRequirement]
) throws -> [ComputeBindingSlot: UInt32] {
    var types: [ComputeBindingSlot: UInt32] = [:]
    for texture in sampled {
        guard !texture.isMultisampled else {
            throw GPUError.unsupported("Vulkan supports single-sampled 2D, array, cube and 3D sampled textures only")
        }
        types[texture.slot] = VK.DescriptorType.sampledImage
    }
    for texture in storage {
        guard texture.dimension != .cube, texture.dimension != .cubeArray, vulkanFormat(texture.format) != nil else {
            throw GPUError.unsupported("Vulkan supports 2D storage textures in core formats only")
        }
        types[texture.slot] = VK.DescriptorType.storageImage
    }
    for sampler in samplers {
        types[sampler.slot] = VK.DescriptorType.sampler
    }
    return types
}

/// Layout bindings of one descriptor set. Binding-array elements share a binding whose count is the array length.
func vulkanLayoutBindings(types: [ComputeBindingSlot: UInt32], group: Int, stages: UInt32) -> [VkDescriptorSetLayoutBinding] {
    var counts: [UInt32: (type: UInt32, count: UInt32)] = [:]
    for (slot, type) in types where Int(slot.group) == group {
        var entry = counts[slot.binding] ?? (type, 0)
        entry.count = max(entry.count, (slot.arrayElement ?? 0) + 1)
        counts[slot.binding] = entry
    }
    return counts.map {
        VkDescriptorSetLayoutBinding(binding: $0.key, descriptorType: $0.value.type, descriptorCount: $0.value.count,
                                     stageFlags: stages)
    }.sorted { $0.binding < $1.binding }
}
