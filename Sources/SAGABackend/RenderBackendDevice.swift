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
                 vertex: RenderStageArtifact, fragment: RenderStageArtifact,
                 nonFilteringSamplers: Set<ComputeBindingSlot> = [], capabilities: DeviceCapabilities? = nil) throws {
        let immediateByteSize = max(vertex.immediateLayout?.wordAlignedByteSize ?? 0,
                                    fragment.immediateLayout?.wordAlignedByteSize ?? 0)
        try capabilities?.requireImmediateSize(immediateByteSize)
        guard sampleCount == 1 || sampleCount == 4 else {
            throw GPUError.validation("Render pipeline sample count must be 1 or 4")
        }
        guard primitive.topology.isStrip || primitive.stripIndexFormat == nil else {
            throw GPUError.validation("A strip index format requires a strip topology")
        }
        if depthState?.format == .depth32FloatStencil8 && capabilities?.features.contains(.depth32FloatStencil8) != true {
            throw GPUError.validation("Feature not advertised")
        }
        if let depthState, depthState.format.isDepthStencil && !depthState.format.hasDepth {
            guard !depthState.writeEnabled, depthState.compare == .always, depthState.bias == .init() else {
                throw GPUError.validation("Stencil-only pipelines require disabled depth state")
            }
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
        if targets.values.contains(where: { $0.colorState.usesSecondBlendSource }) {
            guard additionalColorTargets.isEmpty, fragment.usesDualSourceBlending else {
                throw GPUError.validation("Second-source blend factors require paired outputs and one color target")
            }
        }
        if let components = fragment.dualSourceComponentCount, components != 4 {
            throw GPUError.validation("Dual-source color targets require four output components per source")
        }
        guard fragment.colorOutputKinds.keys.allSatisfy({ targets[$0] != nil }) else {
            throw GPUError.validation("Fragment color output requires a matching color target")
        }
        let bytesPerSample = targets.keys.sorted().reduce(0) { total, location in
            let format = targets[location]!.format
            let alignment = format.colorAttachmentAlignment
            return (total + alignment - 1) / alignment * alignment + format.colorAttachmentByteCost
        }
        guard bytesPerSample <= 32 else {
            throw GPUError.validation("Color attachments exceed the portable bytes-per-sample limit")
        }
        for (location, target) in targets {
            let kind = fragment.colorOutputKinds[location]
            if target.format.isNormalized16Color && capabilities?.features.contains(.normalized16Textures) != true {
                throw GPUError.validation("Feature not advertised")
            }
            if let feature = target.format.hdrRenderFeature, capabilities?.features.contains(feature) != true {
                throw GPUError.validation("Feature not advertised")
            }
            if target.format.isSnorm8Color && capabilities?.features.contains(.snorm8Renderable) != true {
                throw GPUError.validation("Feature not advertised")
            }
            guard target.format.isColorAttachment,
                  kind.map({ target.format.sampleType == TextureSampleType(kind: $0) }) ?? true else {
                throw GPUError.validation("Fragment output type does not match the color attachment format")
            }
            let components = fragment.colorOutputComponentCounts[location] ?? 4
            if kind != nil && components < target.format.colorAttachmentComponentCount {
                throw GPUError.validation("Fragment output has too few components for its color attachment")
            }
            if sampleCount > 1 && !target.format.supportsMultisampledAttachment(in: capabilities) {
                throw GPUError.validation("Color attachment format does not support multisampling")
            }
            guard target.colorState.writeMask.subtracting(.all).isEmpty else {
                throw GPUError.validation("Invalid color write mask")
            }
            if let blend = target.colorState.blend {
                guard (target.format.isNormalizedColor || target.format.isSRGB8Color || target.format.isNormalized16Color ||
                       target.format.isFloat16Color || target.format.isFloat32Color || target.format.isPackedFloatColor ||
                       target.format.isNarrow8UnormColor || target.format.isSnorm8Color || target.format == .rgb9e5Ufloat), kind == .float else {
                    throw GPUError.validation("Blending requires a blendable color attachment with a fragment color output")
                }
                if target.format.isFloat32Color && capabilities?.features.contains(.float32Blendable) != true {
                    throw GPUError.validation("Float32 blending requires float32-blendable")
                }
                if components < 4 && (blend.color.sourceFactor.usesSourceAlpha || blend.color.destinationFactor.usesSourceAlpha) {
                    throw GPUError.validation("Source alpha blending requires a four-component fragment output")
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
            guard (colorFormat.isNormalizedColor || colorFormat.isSRGB8Color || colorFormat == .rgba16Float || colorFormat == .rgb10a2Unorm || colorFormat == .rgba8Snorm ||
                   (colorFormat.isNormalized16Color && colorFormat.colorAttachmentComponentCount == 4)),
                  fragment.colorOutputKind == .float else {
                throw GPUError.validation("Alpha-to-coverage requires a blendable color attachment and a fragment alpha output")
            }
        }
        if fragment.writesDepth && depthState?.format.hasDepth != true {
            throw GPUError.validation("Fragment depth output requires a depth attachment")
        }
        var uniforms: [ComputeBindingSlot: Uniform] = [:]
        for binding in vertex.uniformBindings {
            for (slot, index) in try binding.indexedSlots() {
                uniforms[slot] = .init(slot: slot, minimumByteSize: binding.minimumByteSize,
                                       vertexIndex: index, fragmentIndex: nil, visibility: 1)
            }
        }
        for binding in fragment.uniformBindings {
            for (slot, index) in try binding.indexedSlots() {
                let previous = uniforms[slot]
                uniforms[slot] = .init(slot: slot, minimumByteSize: max(previous?.minimumByteSize ?? 0, binding.minimumByteSize),
                                       vertexIndex: previous?.vertexIndex, fragmentIndex: index,
                                       visibility: (previous?.visibility ?? 0) | 2)
            }
        }
        func merge(
            _ vertex: [ShaderBindingIndex], _ fragment: [ShaderBindingIndex]
        ) -> [HandleBinding] {
            var bindings: [ComputeBindingSlot: HandleBinding] = [:]
            for (slot, index) in vertex {
                bindings[slot] = .init(slot: slot, vertexIndex: index, fragmentIndex: nil, visibility: 1)
            }
            for (slot, index) in fragment {
                let previous = bindings[slot]
                bindings[slot] = .init(slot: slot, vertexIndex: previous?.vertexIndex, fragmentIndex: index,
                                       visibility: (previous?.visibility ?? 0) | 2)
            }
            return bindings.values.sorted {
                ($0.slot.group, $0.slot.binding, $0.slot.arrayElement ?? 0) <
                ($1.slot.group, $1.slot.binding, $1.slot.arrayElement ?? 0)
            }
        }
        var bindingCounts: [IRResourceBinding: UInt32] = [:]
        var layouts: [(IRResourceBinding, ComputeShaderArtifact.BindingArrayLayout?)] = []
        for stage in [vertex, fragment] {
            var bindings = stage.sampledTextures.map { ($0.binding, $0.bindingArray) }
            bindings += stage.samplerBindings.map { ($0.binding, $0.bindingArray) }
            bindings += stage.storageTextures.map { ($0.binding, $0.bindingArray) }
            bindings += stage.storageBindings.map { ($0.binding, $0.bindingArray) }
            bindings += stage.uniformBindings.map { ($0.binding, $0.bindingArray) }
            bindings += stage.externalTextures.map { ($0, nil) }
            layouts += bindings
            for (binding, array) in bindings {
                let count = array?.count ?? 0
                if let previous = bindingCounts.updateValue(count, forKey: binding), previous != count {
                    throw GPUError.validation("Render stages disagree on the binding array length")
                }
            }
        }
        let stageSamplers = try (vertex.samplerBindings + fragment.samplerBindings).flatMap { try $0.requirements() }
        let nonFiltering = try nonFilteringSamplerSlots(nonFilteringSamplers, in: stageSamplers)
        let whollyNonFiltering = whollyNonFilteringSamplerBindings(nonFiltering, in: stageSamplers)
        var sampledRequirements: [ComputeBindingSlot: SampledTextureRequirement] = [:]
        let stageTextures = try (vertex.sampledTextures + fragment.sampledTextures).flatMap {
            try $0.requirements(nonFilteringSamplers: whollyNonFiltering)
        }
        for requirement in stageTextures {
            if let previous = sampledRequirements[requirement.slot] {
                guard previous.sampleType == requirement.sampleType, previous.dimension == requirement.dimension,
                      previous.isMultisampled == requirement.isMultisampled else {
                    throw GPUError.validation("Render stages disagree on the sample type of a texture binding")
                }
                sampledRequirements[requirement.slot] = .init(slot: requirement.slot, sampleType: requirement.sampleType,
                    isMultisampled: requirement.isMultisampled, dimension: requirement.dimension,
                    requiresFiltering: previous.requiresFiltering || requirement.requiresFiltering)
            } else {
                sampledRequirements[requirement.slot] = requirement
            }
        }
        let textureHandles = try merge(vertex.sampledTextures.flatMap { try $0.indexedSlots() },
                                       fragment.sampledTextures.flatMap { try $0.indexedSlots() })
        let textures = textureHandles.map {
            SampledTextureBinding(handle: $0, requirement: sampledRequirements[$0.slot]!)
        }
        var samplerRequirements: [ComputeBindingSlot: SamplerRequirement] = [:]
        for requirement in stageSamplers {
            let previous = samplerRequirements[requirement.slot]
            if let previous, previous.isComparison != requirement.isComparison {
                throw GPUError.validation("Render stages disagree on the type of a sampler binding")
            }
            samplerRequirements[requirement.slot] = .init(
                slot: requirement.slot, isComparison: requirement.isComparison,
                requiresNonFiltering: nonFiltering.contains(requirement.slot))
        }
        let samplerHandles = try merge(vertex.samplerBindings.flatMap { try $0.indexedSlots() },
                                       fragment.samplerBindings.flatMap { try $0.indexedSlots() })
        let samplers = samplerHandles.map {
            SamplerBinding(handle: $0, requirement: samplerRequirements[$0.slot]!)
        }
        var storageRequirements: [ComputeBindingSlot: StorageTextureRequirement] = [:]
        for stage in [vertex, fragment] {
            for requirement in try stage.storageTextures.flatMap({ try $0.requirements() }) {
                guard stage.stage != .vertex || !requirement.writable else {
                    throw GPUError.validation("Writable storage textures require fragment visibility")
                }
                if let previous = storageRequirements[requirement.slot], previous != requirement {
                    let property = previous.dimension == requirement.dimension ? "format or access" : "dimension"
                    throw GPUError.validation("Render stages disagree on the \(property) of a storage texture binding")
                }
                storageRequirements[requirement.slot] = requirement
            }
        }
        let storageHandles = try merge(vertex.storageTextures.flatMap { try $0.indexedSlots() },
                                       fragment.storageTextures.flatMap { try $0.indexedSlots() })
        let storageTextures = storageHandles.map {
            StorageTextureBinding(handle: $0, requirement: storageRequirements[$0.slot]!)
        }
        var storageBufferRequirements: [ComputeBindingSlot: (minimum: UInt64, readOnly: Bool)] = [:]
        for stage in [vertex, fragment] {
            for binding in stage.storageBindings {
                guard stage.stage != .vertex || binding.isReadOnly else {
                    throw GPUError.validation("Writable storage buffers require fragment visibility")
                }
                for (slot, _) in try binding.indexedSlots() {
                    let previous = storageBufferRequirements[slot]
                    if let previous, previous.readOnly != binding.isReadOnly {
                        throw GPUError.validation("Render stages disagree on the access of a storage buffer binding")
                    }
                    storageBufferRequirements[slot] = (max(previous?.minimum ?? 0, binding.minimumByteSize), binding.isReadOnly)
                }
            }
        }
        let bufferHandles = try merge(vertex.storageBindings.flatMap { try $0.indexedSlots() },
                                      fragment.storageBindings.flatMap { try $0.indexedSlots() })
        let storageBuffers = bufferHandles.map { handle in
            let requirement = storageBufferRequirements[handle.slot]!
            return StorageBufferBinding(handle: handle, minimumByteSize: requirement.minimum, isReadOnly: requirement.readOnly)
        }
        let uniformBases = Set(uniforms.keys.map(\.baseSlot))
        let bufferBases = Set(storageBufferRequirements.keys.map(\.baseSlot))
        let textureBases = Set(sampledRequirements.keys.map(\.baseSlot))
        let storageBases = Set(storageRequirements.keys.map(\.baseSlot))
        let samplerBases = Set(samplerRequirements.keys.map(\.baseSlot))
        let externalBindings = merge(vertex.externalTextures.map { (.init(group: $0.group, binding: $0.binding), nil) },
                                     fragment.externalTextures.map { (.init(group: $0.group, binding: $0.binding), nil) })
        let externalSlots = Set(externalBindings.map(\.slot))
        guard textureBases.isDisjoint(with: uniformBases), samplerBases.isDisjoint(with: uniformBases),
              textureBases.isDisjoint(with: samplerBases), storageBases.isDisjoint(with: uniformBases),
              storageBases.isDisjoint(with: textureBases), storageBases.isDisjoint(with: samplerBases),
              bufferBases.isDisjoint(with: uniformBases), bufferBases.isDisjoint(with: textureBases),
              bufferBases.isDisjoint(with: storageBases), bufferBases.isDisjoint(with: samplerBases),
              externalSlots.isDisjoint(with: uniformBases), externalSlots.isDisjoint(with: bufferBases),
              externalSlots.isDisjoint(with: textureBases), externalSlots.isDisjoint(with: storageBases),
              externalSlots.isDisjoint(with: samplerBases) else {
            throw GPUError.validation("Render stages disagree on the resource type of a binding")
        }
        self.init(id: id, colorFormat: colorFormat, vertexBuffers: vertexBuffers, depthState: depthState,
                  sampleCount: sampleCount, sampleMask: sampleMask, alphaToCoverageEnabled: alphaToCoverageEnabled,
                  writesColor: fragment.colorOutputKind != nil, primitive: primitive,
                  colorState: colorState, additionalColorTargets: additionalColorTargets,
                  additionalColorOutputs: Set(fragment.colorOutputKinds.keys.filter { $0 != 0 }), uniforms: uniforms.values.sorted {
                      ($0.slot.group, $0.slot.binding, $0.slot.arrayElement ?? 0) <
                      ($1.slot.group, $1.slot.binding, $1.slot.arrayElement ?? 0)
                  },
                  storageBuffers: storageBuffers, sampledTextures: textures,
                  storageTextures: storageTextures, samplerBindings: samplers, externalTextureBindings: externalBindings,
                  webBindingIndices: try resolveWebBindingIndices(layouts), immediateByteSize: immediateByteSize)
    }
}
