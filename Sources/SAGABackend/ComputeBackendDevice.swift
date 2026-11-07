import SAGATypes
import SAGAShaderIR

/// Backend-owned native pipelines live until device teardown or allocation rollback.
public protocol ComputeBackendDevice: CommandBackendDevice {
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) async throws
    func destroyComputePipeline(id: ResourceID) async
}

extension ComputePipeline {
    package init(id: ResourceID, shader: ComputeShaderArtifact) throws {
        self.init(id: id, storageBindings: shader.storageBindings.map {
            .init(slot: .init(group: $0.binding.group, binding: $0.binding.binding),
                  minimumByteSize: $0.minimumByteSize, isReadOnly: $0.isReadOnly)
        }, uniformBindings: shader.uniformBindings.map {
            .init(slot: .init(group: $0.binding.group, binding: $0.binding.binding),
                  minimumByteSize: $0.minimumByteSize, isReadOnly: true, usage: .uniform)
        }, textureBindings: shader.sampledTextures.map(\.requirement),
        storageTextureBindings: try shader.storageTextures.map {
            try $0.requirement()
        }, samplerBindings: shader.samplerBindings.map(\.requirement))
    }
}

extension ComputeShaderArtifact.SamplerBinding {
    package var requirement: SamplerRequirement {
        .init(slot: .init(group: binding.group, binding: binding.binding), isComparison: isComparison,
              requiresNonFiltering: requiresNonFiltering)
    }
}

extension ComputeShaderArtifact.SampledTextureBinding {
    package var requirement: SampledTextureRequirement {
        let type = isDepth ? TextureSampleType.depth : TextureSampleType(kind: sampleKind)
        let view: TextureViewDimension = switch dimension {
        case .cube: isArrayed ? .cubeArray : .cube
        case .d3: .d3
        default: isArrayed ? .d2Array : .d2
        }
        return .init(slot: .init(group: binding.group, binding: binding.binding), sampleType: type,
                     isMultisampled: isMultisampled, dimension: view)
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
    package func requirement() throws -> StorageTextureRequirement {
        guard let mode = StorageTextureAccess(rawValue: UInt32(access.rawValue)) else {
            throw GPUError.validation("Unsupported storage texture access")
        }
        return .init(slot: .init(group: binding.group, binding: binding.binding), format: try textureFormat(),
                     access: mode, isArrayed: isArrayed)
    }

    package func textureFormat() throws -> TextureFormat {
        switch format {
        case .rgba8Unorm: .rgba8Unorm
        case .rgba8Snorm: .rgba8Snorm
        case .rgba8Uint: .rgba8Uint
        case .rgba8Sint: .rgba8Sint
        case .rgba16Uint: .rgba16Uint
        case .rgba16Sint: .rgba16Sint
        case .rgba16Float: .rgba16Float
        case .r32Uint: .r32Uint
        case .r32Sint: .r32Sint
        case .r32Float: .r32Float
        case .rg32Uint: .rg32Uint
        case .rg32Sint: .rg32Sint
        case .rg32Float: .rg32Float
        case .rgba32Uint: .rgba32Uint
        case .rgba32Sint: .rgba32Sint
        case .rgba32Float: .rgba32Float
        default: throw GPUError.validation("Unsupported storage texture format")
        }
    }
}

extension Sequence where Element == ComputeShaderArtifact.StorageTextureBinding {
    package var requiredStorageTextureFeatures: Set<GPUFeature> {
        var features: Set<GPUFeature> = []
        for binding in self where binding.access.contains(.load) {
            features.insert(binding.access.contains(.store) ? .storageTextureReadWrite : .storageTextureRead)
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
