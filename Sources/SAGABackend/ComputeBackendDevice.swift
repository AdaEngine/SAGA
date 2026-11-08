import SAGATypes
import SAGAShaderIR

/// Backend-owned native pipelines live until device teardown or allocation rollback.
public protocol ComputeBackendDevice: CommandBackendDevice {
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) async throws
    func destroyComputePipeline(id: ResourceID) async
}

package typealias ShaderBindingIndex = (slot: ComputeBindingSlot, index: UInt32?)

extension ComputePipeline {
    package init(id: ResourceID, shader: ComputeShaderArtifact, nonFilteringSamplers: Set<ComputeBindingSlot> = []) throws {
        let maximumLinearWorkgroupCount: UInt32?
        if shader.usesGlobalInvocationIndex {
            let size = shader.workgroupSize
            guard size.x > 0, size.y > 0, size.z > 0 else {
                throw GPUError.validation("Linear indexing requires a nonempty workgroup")
            }
            let limit = UInt32.max / size.x / size.y / size.z
            guard limit > 0 else { throw GPUError.validation("Linear indexing workgroup exceeds u32") }
            maximumLinearWorkgroupCount = limit
        } else {
            maximumLinearWorkgroupCount = shader.usesWorkgroupIndex ? .max : nil
        }
        let samplers = try shader.samplerBindings.flatMap { try $0.requirements() }
        let nonFiltering = try nonFilteringSamplerSlots(nonFilteringSamplers, in: samplers)
        let whollyNonFiltering = whollyNonFilteringSamplerBindings(nonFiltering, in: samplers)
        var layouts = shader.sampledTextures.map { ($0.binding, $0.bindingArray) }
        layouts += shader.samplerBindings.map { ($0.binding, $0.bindingArray) }
        layouts += shader.storageTextures.map { ($0.binding, $0.bindingArray) }
        layouts += shader.storageBindings.map { ($0.binding, $0.bindingArray) }
        layouts += shader.uniformBindings.map { ($0.binding, $0.bindingArray) }
        layouts += shader.externalTextures.map { ($0, nil) }
        self.init(id: id, storageBindings: try shader.storageBindings.flatMap { buffer in
            try buffer.binding.computeSlots(array: buffer.bindingArray, limit: ComputeShaderArtifact.storageBindingCapacity).map {
                .init(slot: $0, minimumByteSize: buffer.minimumByteSize, isReadOnly: buffer.isReadOnly)
            }
        }, uniformBindings: try shader.uniformBindings.flatMap { buffer in
            try buffer.binding.computeSlots(array: buffer.bindingArray, limit: ComputeShaderArtifact.uniformBindingCapacity).map {
                .init(slot: $0, minimumByteSize: buffer.minimumByteSize, isReadOnly: true, usage: .uniform)
            }
        }, textureBindings: try shader.sampledTextures.flatMap { binding in
            try binding.requirements(nonFilteringSamplers: whollyNonFiltering)
        }, storageTextureBindings: try shader.storageTextures.flatMap { binding in
            try binding.requirements()
        }, samplerBindings: samplers.map {
            .init(slot: $0.slot, isComparison: $0.isComparison, requiresNonFiltering: nonFiltering.contains($0.slot))
        }, externalTextureBindings: shader.externalTextures.map { .init(group: $0.group, binding: $0.binding) },
        webBindingIndices: try resolveWebBindingIndices(layouts),
        immediateByteSize: shader.immediateLayout?.wordAlignedByteSize ?? 0,
        maximumLinearWorkgroupCount: maximumLinearWorkgroupCount,
        requiredSubgroupSize: shader.requiredSubgroupSize)
    }
}

func resolveWebBindingIndices(
    _ layouts: [(IRResourceBinding, ComputeShaderArtifact.BindingArrayLayout?)]
) throws -> [ComputeBindingSlot: UInt32] {
    guard layouts.contains(where: { $0.1?.webBindingBase != nil }) else { return [:] }
    var indices: [ComputeBindingSlot: UInt32] = [:]
    var owners: [IRResourceBinding: ComputeBindingSlot] = [:]
    let limit = ComputeShaderArtifact.portableBindingIndexLimit
    for (binding, array) in layouts {
        let base: UInt32
        if let array {
            guard let webBase = array.webBindingBase, webBase < limit, array.count <= limit - webBase else {
                throw GPUError.validation("Invalid WebGPU binding array layout")
            }
            base = webBase
        } else {
            base = binding.binding
        }
        let slots = try binding.computeSlots(array: array, limit: limit)
        for (offset, slot) in slots.enumerated() {
            let index = base + UInt32(offset)
            let physical = IRResourceBinding(group: slot.group, binding: index)
            if let owner = owners.updateValue(slot, forKey: physical), owner != slot {
                throw GPUError.validation("Overlapping physical WebGPU bindings")
            }
            if let previous = indices.updateValue(index, forKey: slot), previous != index {
                throw GPUError.validation("Render stages disagree on the physical WebGPU binding")
            }
        }
    }
    return indices.filter { $0.key.arrayElement != nil }
}

extension ComputeShaderArtifact.StorageBinding {
    package func indexedSlots() throws -> [ShaderBindingIndex] {
        try binding.indexedSlots(array: bindingArray, limit: ComputeShaderArtifact.storageBindingCapacity, first: flatBindingIndex)
    }
}

extension ComputeShaderArtifact.UniformBinding {
    package func indexedSlots() throws -> [ShaderBindingIndex] {
        try binding.indexedSlots(array: bindingArray, limit: ComputeShaderArtifact.uniformBindingCapacity, first: flatBindingIndex)
    }
}

extension IRResourceBinding {
    package func computeSlots(array: ComputeShaderArtifact.BindingArrayLayout?, limit: UInt32) throws -> [ComputeBindingSlot] {
        guard let array else { return [.init(group: group, binding: binding)] }
        guard array.count > 0, array.count <= limit else {
            throw GPUError.validation("Invalid resource binding array length")
        }
        return (0..<array.count).map { .init(group: group, binding: binding, arrayElement: $0) }
    }

    package func indexedSlots(array: ComputeShaderArtifact.BindingArrayLayout?, limit: UInt32, first: UInt32?)
        throws -> [ShaderBindingIndex]
    {
        let slots = try computeSlots(array: array, limit: limit)
        return try slots.enumerated().map { offset, slot in
            guard let first else { return (slot, nil) }
            let (index, overflow) = first.addingReportingOverflow(UInt32(offset))
            guard !overflow else { throw GPUError.validation("Resource binding index overflow") }
            return (slot, index)
        }
    }
}

func nonFilteringSamplerSlots(
    _ selected: Set<ComputeBindingSlot>, in samplers: [SamplerRequirement]
) throws -> Set<ComputeBindingSlot> {
    let ordinary = Set(samplers.filter { !$0.isComparison }.map(\.slot))
    let comparison = Set(samplers.filter(\.isComparison).map(\.slot))
    guard selected.isSubset(of: ordinary), selected.isDisjoint(with: comparison) else {
        throw GPUError.validation("Non-filtering selection must name a non-comparison sampler binding")
    }
    return selected.union(samplers.filter(\.requiresNonFiltering).map(\.slot))
}

func whollyNonFilteringSamplerBindings(
    _ nonFiltering: Set<ComputeBindingSlot>, in samplers: [SamplerRequirement]
) -> Set<ComputeBindingSlot> {
    let slots = samplers.map(\.slot)
    return Set(slots.map(\.baseSlot)).subtracting(slots.filter { !nonFiltering.contains($0) }.map(\.baseSlot))
}

extension ComputeShaderArtifact.SamplerBinding {
    package func indexedSlots() throws -> [ShaderBindingIndex] {
        try binding.indexedSlots(array: bindingArray, limit: ComputeShaderArtifact.samplerBindingCapacity, first: flatBindingIndex)
    }

    package var requirement: SamplerRequirement {
        .init(slot: .init(group: binding.group, binding: binding.binding), isComparison: isComparison,
              requiresNonFiltering: requiresNonFiltering)
    }

    func requirements() throws -> [SamplerRequirement] {
        try binding.computeSlots(array: bindingArray, limit: ComputeShaderArtifact.samplerBindingCapacity).map { slot in
            .init(slot: slot, isComparison: isComparison, requiresNonFiltering: requiresNonFiltering)
        }
    }
}

extension ComputeShaderArtifact.SampledTextureBinding {
    package func indexedSlots() throws -> [ShaderBindingIndex] {
        try binding.indexedSlots(array: bindingArray, limit: ComputeShaderArtifact.sampledTextureBindingCapacity, first: flatBindingIndex)
    }

    func requirements(nonFilteringSamplers: Set<ComputeBindingSlot>) throws -> [SampledTextureRequirement] {
        try binding.computeSlots(array: bindingArray, limit: ComputeShaderArtifact.sampledTextureBindingCapacity).map {
            requirement(nonFilteringSamplers: nonFilteringSamplers, slot: $0)
        }
    }

    func requirement(nonFilteringSamplers: Set<ComputeBindingSlot>, slot: ComputeBindingSlot? = nil) -> SampledTextureRequirement {
        let type = isDepth ? TextureSampleType.depth : TextureSampleType(kind: sampleKind)
        let view: TextureViewDimension = switch dimension {
        case .cube: isArrayed ? .cubeArray : .cube
        case .d3: .d3
        case .d1: .d1
        default: isArrayed ? .d2Array : .d2
        }
        let requiresFiltering = samplingSamplers.contains {
            !nonFilteringSamplers.contains(.init(group: $0.group, binding: $0.binding))
        }
        return .init(slot: slot ?? .init(group: binding.group, binding: binding.binding), sampleType: type,
                     isMultisampled: isMultisampled, dimension: view, requiresFiltering: requiresFiltering)
    }
}

extension TextureSampleType {
    package init(kind: IRScalarKind) {
        switch kind {
        case .sint: self = .sint
        case .uint: self = .uint
        default: self = .float
        }
    }
}

extension ComputeShaderArtifact.StorageTextureBinding {
    package func indexedSlots() throws -> [ShaderBindingIndex] {
        try binding.indexedSlots(array: bindingArray, limit: ComputeShaderArtifact.storageTextureBindingCapacity, first: flatBindingIndex)
    }

    func requirements() throws -> [StorageTextureRequirement] {
        try binding.computeSlots(array: bindingArray, limit: ComputeShaderArtifact.storageTextureBindingCapacity).map {
            try requirement(slot: $0)
        }
    }

    package func requirement(slot: ComputeBindingSlot? = nil) throws -> StorageTextureRequirement {
        guard let mode = StorageTextureAccess(rawValue: UInt32(access.rawValue)) else {
            throw GPUError.validation("Unsupported storage texture access")
        }
        let view: TextureViewDimension = switch dimension {
        case .d1: .d1
        case .d3: .d3
        default: isArrayed ? .d2Array : .d2
        }
        return .init(slot: slot ?? .init(group: binding.group, binding: binding.binding), format: textureFormat(),
                     access: mode, dimension: view)
    }

    package func textureFormat() -> TextureFormat {
        switch format {
        case .bgra8Unorm: .bgra8Unorm
        case .rgba8Unorm: .rgba8Unorm
        case .rgba8Snorm: .rgba8Snorm
        case .rgba8Uint: .rgba8Uint
        case .rgba8Sint: .rgba8Sint
        case .rgba16Uint: .rgba16Uint
        case .rgba16Sint: .rgba16Sint
        case .rgba16Float: .rgba16Float
        case .r32Uint: .r32Uint
        case .r64Uint: .r64Uint
        case .r32Sint: .r32Sint
        case .r8Unorm: .r8Unorm
        case .r8Snorm: .r8Snorm
        case .r8Uint: .r8Uint
        case .r8Sint: .r8Sint
        case .rg8Unorm: .rg8Unorm
        case .rg8Snorm: .rg8Snorm
        case .rg8Uint: .rg8Uint
        case .rg8Sint: .rg8Sint
        case .rgb10a2Unorm: .rgb10a2Unorm
        case .rgb10a2Uint: .rgb10a2Uint
        case .rg11b10Ufloat: .rg11b10Ufloat
        case .r16Unorm: .r16Unorm
        case .rg16Unorm: .rg16Unorm
        case .rgba16Unorm: .rgba16Unorm
        case .r16Snorm: .r16Snorm
        case .rg16Snorm: .rg16Snorm
        case .rgba16Snorm: .rgba16Snorm
        case .r16Uint: .r16Uint
        case .r16Sint: .r16Sint
        case .rg16Uint: .rg16Uint
        case .rg16Sint: .rg16Sint
        case .r16Float: .r16Float
        case .rg16Float: .rg16Float
        case .r32Float: .r32Float
        case .rg32Uint: .rg32Uint
        case .rg32Sint: .rg32Sint
        case .rg32Float: .rg32Float
        case .rgba32Uint: .rgba32Uint
        case .rgba32Sint: .rgba32Sint
        case .rgba32Float: .rgba32Float
        }
    }
}

extension Sequence where Element == ComputeShaderArtifact.StorageTextureBinding {
    package var requiredStorageTextureFeatures: Set<GPUFeature> {
        var features: Set<GPUFeature> = []
        for binding in self {
            if binding.format == .r64Uint {
                features.insert(.textureInt64Atomic)
                continue
            }
            if binding.access.contains(.atomic) {
                features.insert(.textureAtomic)
                continue
            }
            switch binding.format {
            case .r16Float, .rg16Float:
                features.insert(.halfFloatStorageTextures)
            case .r16Uint, .r16Sint, .rg16Uint, .rg16Sint:
                features.insert(.narrowInteger16StorageTextures)
            default:
                break
            }
            if binding.format.isNormalized16Bit { features.insert(.normalized16Textures) }
            if binding.format.isPackedColor { features.insert(.packedColorStorageTextures) }
            if binding.format.isNarrow8Bit { features.insert(.narrow8BitStorageTextures) }
            if binding.format == .bgra8Unorm { features.insert(.bgra8UnormStorage) }
            if binding.access.contains([.load, .store]) {
                if binding.format.requires8BitReadWriteFeature { features.insert(.storage8BitReadWrite) }
                if binding.format.requiresWideRGBAReadWriteFeature { features.insert(.wideRGBAStorageReadWrite) }
                switch binding.format {
                case .r16Float:
                    features.insert(.r16FloatStorageReadWrite)
                case .r16Uint, .r16Sint, .rgba16Uint, .rgba16Sint:
                    features.insert(.integer16StorageReadWrite)
                default:
                    break
                }
            }
            if binding.access.contains(.load) {
                features.insert(binding.access.contains(.store) ? .storageTextureReadWrite : .storageTextureRead)
            }
        }
        return features
    }
}

extension ComputeShaderArtifact {
    package var storageTextureFeatures: Set<GPUFeature> {
        var features = storageTextures.requiredStorageTextureFeatures
        if usesTextureBarrier { features.insert(.storageTextureRead) }
        return features
    }
}
