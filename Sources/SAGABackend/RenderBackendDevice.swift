import SAGATypes
import SAGAShaderIR

public protocol RenderBackendDevice: CommandBackendDevice {
    func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws
    func destroyTexture(id: ResourceID) async
    func allocateRenderPipeline(_ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact) async throws
    func destroyRenderPipeline(id: ResourceID) async
}

public protocol TextureViewBackendDevice: RenderBackendDevice {
    func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>) async throws
    func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>,
                             aspect: TextureViewAspect) async throws
    func allocateTextureView(id: ResourceID, texture: ResourceID, descriptor: TextureViewDescriptor) async throws
    func destroyTextureView(id: ResourceID) async
}

extension TextureViewBackendDevice {
    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>,
                                    aspect: TextureViewAspect) async throws {
        guard aspect == .all else { throw GPUError.unsupported("Backend does not implement texture aspect views") }
        try await allocateTextureView(id: id, texture: texture, mipLevels: mipLevels)
    }

    public func allocateTextureView(id: ResourceID, texture: ResourceID, descriptor: TextureViewDescriptor) async throws {
        guard descriptor.baseArrayLayer == 0, descriptor.arrayLayerCount == nil || descriptor.arrayLayerCount == 1,
              descriptor.dimension == nil || descriptor.dimension == .d2,
              let count = descriptor.mipLevelCount, count > 0,
              count <= UInt32.max - descriptor.baseMipLevel else {
            throw GPUError.unsupported("Backend does not implement layer-range texture views")
        }
        let levels = descriptor.baseMipLevel..<(descriptor.baseMipLevel + count)
        try await allocateTextureView(id: id, texture: texture, mipLevels: levels, aspect: descriptor.aspect)
    }
}

extension RenderPipeline {
    package init(id: ResourceID, colorFormat: TextureFormat, vertexBuffers: [VertexBufferLayout],
                 depthState: RenderDepthState? = nil, sampleCount: UInt32 = 1, sampleMask: UInt32 = .max,
                 alphaToCoverageEnabled: Bool = false, primitive: RenderPrimitiveState = .init(), colorState: RenderColorState = .init(),
                 additionalColorTargets: [UInt32: RenderColorTarget] = [:],
                 vertex: RenderStageArtifact, fragment: RenderStageArtifact) throws {
        guard sampleCount == 1 || sampleCount == 4 else {
            throw GPUError.validation("Render pipeline sample count must be 1 or 4")
        }
        guard primitive.topology.isStrip || primitive.stripIndexFormat == nil else {
            throw GPUError.validation("A strip index format requires a strip topology")
        }
        if let depthState, depthState.stencil != nil && !depthState.format.hasStencil {
            throw GPUError.validation("Stencil state requires a stencil attachment format")
        }
        if let bias = depthState?.bias {
            guard bias.slopeScale.isFinite, bias.clamp.isFinite else {
                throw GPUError.validation("Depth bias slope scale and clamp must be finite")
            }
            guard bias == .init() || primitive.topology == .triangleList || primitive.topology == .triangleStrip else {
                throw GPUError.validation("Depth bias requires a triangle topology")
            }
        }
        guard additionalColorTargets.keys.allSatisfy({ (1..<8).contains($0) }) else {
            throw GPUError.validation("Additional color target locations must be in 1..<8")
        }
        var targets = additionalColorTargets
        targets[0] = .init(format: colorFormat, colorState: colorState)
        guard fragment.colorOutputKinds.keys.allSatisfy({ targets[$0] != nil }) else {
            throw GPUError.validation("Fragment color output requires a matching color target")
        }
        // All supported color formats have byte alignment one in the WebGPU color budget.
        let bytesPerSample = targets.values.reduce(0) { $0 + ($1.format.isNormalizedColor ? 8 : 4) }
        guard bytesPerSample <= 32 else {
            throw GPUError.validation("Color attachments exceed the portable bytes-per-sample limit")
        }
        for (location, target) in targets {
            let kind = fragment.colorOutputKinds[location]
            guard target.format.isColorAttachment,
                  kind.map({ target.format.sampleType == TextureSampleType(kind: $0) }) ?? true else {
                throw GPUError.validation("Fragment output type does not match the color attachment format")
            }
            guard target.colorState.writeMask.subtracting(.all).isEmpty else {
                throw GPUError.validation("Invalid color write mask")
            }
            if let blend = target.colorState.blend {
                guard target.format.isNormalizedColor, kind == .float else {
                    throw GPUError.validation("Blending requires a blendable color attachment with a fragment color output")
                }
                guard blend.color.hasValidFactors, blend.alpha.hasValidFactors else {
                    throw GPUError.validation("Min and max blending require one for both factors")
                }
            }
        }
        if alphaToCoverageEnabled {
            guard sampleCount > 1 else {
                throw GPUError.validation("Alpha-to-coverage requires a multisampled pipeline")
            }
            guard !fragment.writesSampleMask else {
                throw GPUError.validation("Alpha-to-coverage is incompatible with fragment sample_mask output")
            }
            guard colorFormat.isNormalizedColor, fragment.colorOutputKind == .float else {
                throw GPUError.validation("Alpha-to-coverage requires a blendable color attachment and a fragment alpha output")
            }
        }
        if fragment.writesDepth && depthState == nil {
            throw GPUError.validation("Fragment depth output requires a depth attachment")
        }
        var uniforms: [ComputeBindingSlot: Uniform] = [:]
        for binding in vertex.uniformBindings {
            let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
            uniforms[slot] = .init(slot: slot, minimumByteSize: binding.minimumByteSize,
                                   vertexIndex: binding.flatBindingIndex, fragmentIndex: nil, visibility: 1)
        }
        for binding in fragment.uniformBindings {
            let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
            let previous = uniforms[slot]
            uniforms[slot] = .init(slot: slot, minimumByteSize: max(previous?.minimumByteSize ?? 0, binding.minimumByteSize),
                                   vertexIndex: previous?.vertexIndex, fragmentIndex: binding.flatBindingIndex,
                                   visibility: (previous?.visibility ?? 0) | 2)
        }
        func merge(
            _ vertex: [(IRResourceBinding, UInt32?)], _ fragment: [(IRResourceBinding, UInt32?)]
        ) -> [HandleBinding] {
            var bindings: [ComputeBindingSlot: HandleBinding] = [:]
            for (binding, index) in vertex {
                let slot = ComputeBindingSlot(group: binding.group, binding: binding.binding)
                bindings[slot] = .init(slot: slot, vertexIndex: index, fragmentIndex: nil, visibility: 1)
            }
            for (binding, index) in fragment {
                let slot = ComputeBindingSlot(group: binding.group, binding: binding.binding)
                let previous = bindings[slot]
                bindings[slot] = .init(slot: slot, vertexIndex: previous?.vertexIndex, fragmentIndex: index,
                                       visibility: (previous?.visibility ?? 0) | 2)
            }
            return bindings.values.sorted { ($0.slot.group, $0.slot.binding) < ($1.slot.group, $1.slot.binding) }
        }
        var sampledRequirements: [ComputeBindingSlot: SampledTextureRequirement] = [:]
        for binding in vertex.sampledTextures + fragment.sampledTextures {
            let requirement = binding.requirement
            if let previous = sampledRequirements[requirement.slot], previous != requirement {
                throw GPUError.validation("Render stages disagree on the sample type of a texture binding")
            }
            sampledRequirements[requirement.slot] = requirement
        }
        let textureHandles = merge(vertex.sampledTextures.map { ($0.binding, $0.flatBindingIndex) },
                                   fragment.sampledTextures.map { ($0.binding, $0.flatBindingIndex) })
        let textures = textureHandles.map {
            SampledTextureBinding(handle: $0, requirement: sampledRequirements[$0.slot]!)
        }
        var samplerRequirements: [ComputeBindingSlot: SamplerRequirement] = [:]
        for binding in vertex.samplerBindings + fragment.samplerBindings {
            let requirement = binding.requirement
            let previous = samplerRequirements[requirement.slot]
            if let previous, previous.isComparison != requirement.isComparison {
                throw GPUError.validation("Render stages disagree on the type of a sampler binding")
            }
            samplerRequirements[requirement.slot] = .init(
                slot: requirement.slot, isComparison: requirement.isComparison,
                requiresNonFiltering: requirement.requiresNonFiltering || previous?.requiresNonFiltering == true)
        }
        let samplerHandles = merge(vertex.samplerBindings.map { ($0.binding, $0.flatBindingIndex) },
                                   fragment.samplerBindings.map { ($0.binding, $0.flatBindingIndex) })
        let samplers = samplerHandles.map {
            SamplerBinding(handle: $0, requirement: samplerRequirements[$0.slot]!)
        }
        var storageRequirements: [ComputeBindingSlot: StorageTextureRequirement] = [:]
        for stage in [vertex, fragment] {
            for binding in stage.storageTextures {
                let requirement = try binding.requirement()
                guard stage.stage != .vertex || !requirement.writable else {
                    throw GPUError.validation("Writable storage textures require fragment visibility")
                }
                if let previous = storageRequirements[requirement.slot], previous != requirement {
                    let property = previous.isArrayed == requirement.isArrayed ? "format or access" : "dimension"
                    throw GPUError.validation("Render stages disagree on the \(property) of a storage texture binding")
                }
                storageRequirements[requirement.slot] = requirement
            }
        }
        let storageHandles = merge(vertex.storageTextures.map { ($0.binding, $0.flatBindingIndex) },
                                   fragment.storageTextures.map { ($0.binding, $0.flatBindingIndex) })
        let storageTextures = storageHandles.map {
            StorageTextureBinding(handle: $0, requirement: storageRequirements[$0.slot]!)
        }
        let textureSlots = Set(sampledRequirements.keys)
        let storageSlots = Set(storageRequirements.keys)
        let samplerSlots = Set(samplerRequirements.keys)
        guard textureSlots.isDisjoint(with: uniforms.keys), samplerSlots.isDisjoint(with: uniforms.keys),
              textureSlots.isDisjoint(with: samplerSlots), storageSlots.isDisjoint(with: uniforms.keys),
              storageSlots.isDisjoint(with: textureSlots), storageSlots.isDisjoint(with: samplerSlots) else {
            throw GPUError.validation("Render stages disagree on the resource type of a binding")
        }
        self.init(id: id, colorFormat: colorFormat, vertexBuffers: vertexBuffers, depthState: depthState,
                  sampleCount: sampleCount, sampleMask: sampleMask, alphaToCoverageEnabled: alphaToCoverageEnabled,
                  writesColor: fragment.colorOutputKind != nil, primitive: primitive,
                  colorState: colorState, additionalColorTargets: additionalColorTargets,
                  additionalColorOutputs: Set(fragment.colorOutputKinds.keys.filter { $0 != 0 }), uniforms: uniforms.values.sorted { ($0.slot.group, $0.slot.binding) < ($1.slot.group, $1.slot.binding) },
                  sampledTextures: textures, storageTextures: storageTextures, samplerBindings: samplers)
    }
}
