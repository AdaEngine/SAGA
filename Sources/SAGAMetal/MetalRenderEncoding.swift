import SAGABackend
import SAGAShaderIR
import SAGATypes
#if canImport(Metal)
import Metal

extension MetalDevice {
    func nativeTexture(_ id: ResourceID) -> any MTLTexture {
        textureViews[id]?.native ?? textures[id]!.native
    }

    func renderAttachmentLayers(_ id: ResourceID, depthSlice: UInt32? = nil) -> Range<UInt32> {
        if let depthSlice { return depthSlice..<(depthSlice + 1) }
        return textureViews[id]?.resource.layers ?? 0..<textures[id]!.descriptor.arrayLayerCount
    }

    private func markRenderedTexture(_ id: ResourceID, level: UInt32, depthSlice: UInt32? = nil,
                                     initialized: inout [ResourceID: Set<TextureSubresource>]) {
        let view = textureViews[id]?.resource
        for layer in renderAttachmentLayers(id, depthSlice: depthSlice) {
            initialized[view?.texture ?? id, default: []].insert(.init(level: view?.levels.lowerBound ?? level, layer: layer))
        }
    }

    func markRenderedAttachments(
        target: ResourceID?, level: UInt32, depthSlice: UInt32?, depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?,
        additionalColors: [UInt32: RenderColorAttachment], initialized: inout [ResourceID: Set<TextureSubresource>]
    ) {
        if let target { markRenderedTexture(target, level: level, depthSlice: depthSlice, initialized: &initialized) }
        if let resolve { markRenderedTexture(resolve.texture, level: resolve.mipLevel, initialized: &initialized) }
        for color in additionalColors.values {
            markRenderedTexture(color.texture, level: color.mipLevel, depthSlice: color.depthSlice, initialized: &initialized)
            if let resolve = color.resolve {
                markRenderedTexture(resolve.texture, level: resolve.mipLevel, initialized: &initialized)
            }
        }
        if let depth { markRenderedTexture(depth.texture, level: depth.mipLevel, initialized: &initialized) }
    }

    func initializeLoadedAttachments(
        target: ResourceID?, level: UInt32, depthSlice: UInt32?, colorLoad: RenderColorLoadAction,
        depth: RenderDepthAttachment?, additionalColors: [UInt32: RenderColorAttachment], on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        func initialize(_ id: ResourceID, level: UInt32, depthSlice: UInt32? = nil) throws {
            let view = textureViews[id]?.resource
            for layer in renderAttachmentLayers(id, depthSlice: depthSlice) {
                try initializeTexture(view?.texture ?? id, level: view?.levels.lowerBound ?? level,
                                      layer: layer, on: commands, initialized: &initialized)
            }
        }
        if let target, case .load = colorLoad { try initialize(target, level: level, depthSlice: depthSlice) }
        for attachment in additionalColors.values where attachment.loadAction == .load {
            try initialize(attachment.texture, level: attachment.mipLevel, depthSlice: attachment.depthSlice)
        }
        if let depth {
            let loadsDepth = depth.loadAction == .load || depth.loadAction == .readOnly
            let loadsStencil = depth.stencil?.loadAction == .load || depth.stencil?.loadAction == .readOnly
            if loadsDepth || loadsStencil { try initialize(depth.texture, level: depth.mipLevel) }
        }
    }

    func recordNativePass(
        _ pass: NativePass, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws -> [@MainActor @Sendable () -> Void] {
        var nativeBuffers: [ResourceID: any MTLBuffer] = [:]
        var nativeTextures: [ResourceID: any MTLTexture] = [:]
        for use in pass.uses {
            let id = use.resource
            if let buffer = buffers[id] {
                nativeBuffers[id] = buffer.native
            } else if let texture = textures[id] {
                // A native write may touch only part of a subresource.
                try initializeTextureSubresources(id, on: commands, initialized: &initialized)
                nativeTextures[id] = texture.native
            } else {
                throw GPUError.validation("A native pass names an unknown Metal resource")
            }
        }
        let context = MetalNativePassContext(commandBuffer: commands, buffers: nativeBuffers, textures: nativeTextures)
        try pass.recordOnMainActor(into: context)
        return context.committedHandlers
    }

    func validateExternalTextureLayouts(
        _ layouts: [ComputeShaderArtifact.ExternalTextureLayout], bindings: [IRResourceBinding],
        textureIndices: Set<UInt32>, uniformIndices: Set<UInt32>
    ) throws {
        guard layouts.count == bindings.count, Set(layouts.map(\.binding)) == Set(bindings) else {
            throw GPUError.validation("Missing Metal external texture layout")
        }
        var textures = textureIndices
        var uniforms = uniformIndices
        let base = ComputeShaderArtifact.uniformBindingBase
        let uniformRange = base..<(base + ComputeShaderArtifact.uniformBindingCapacity)
        for layout in layouts {
            guard layout.planeBindingIndices.count == 3,
                  layout.planeBindingIndices.allSatisfy({
                      $0 < ComputeShaderArtifact.sampledTextureBindingCapacity && textures.insert($0).inserted
                  }),
                  uniformRange.contains(layout.parametersBindingIndex),
                  uniforms.insert(layout.parametersBindingIndex).inserted else {
                throw GPUError.validation("Invalid Metal external texture layout")
            }
        }
    }

    func bindExternalTextures(
        _ layouts: [ComputeShaderArtifact.ExternalTextureLayout], bindings: [ComputeBindingSlot: ResourceID],
        setPlane: (any MTLTexture, Int) -> Void, setParameters: (UnsafeRawPointer, Int, Int) -> Void
    ) {
        for layout in layouts {
            let slot = ComputeBindingSlot(group: layout.binding.group, binding: layout.binding.binding)
            let texture = externalTextures[bindings[slot]!]!
            for (index, nativeSlot) in layout.planeBindingIndices.enumerated() {
                let plane = texture.planes[index < texture.planes.count ? index : 0]
                setPlane(nativeTexture(plane), Int(nativeSlot))
            }
            texture.parameters.withUnsafeBytes {
                setParameters($0.baseAddress!, $0.count, Int(layout.parametersBindingIndex))
            }
        }
    }

    func runtimeLengthRequirements(
        _ bindings: [ComputeShaderArtifact.StorageBinding]
    ) throws -> [(ComputeBindingSlot, UInt32, UInt32)] {
        try bindings.flatMap { binding -> [(ComputeBindingSlot, UInt32, UInt32)] in
            guard let stride = binding.runtimeArrayStride else { return [] }
            return try binding.binding.computeSlots(array: binding.bindingArray, limit: ComputeShaderArtifact.storageBindingCapacity)
                .map { ($0, stride, binding.runtimeArrayOffset) }
        }
    }

    func runtimeLengthValues(
        _ requirements: [(ComputeBindingSlot, UInt32, UInt32)], bindings: [ComputeBindingSlot: ComputeBufferBinding]
    ) throws -> [UInt32] {
        try requirements.map { slot, stride, offset in
            guard let binding = bindings[slot], let descriptor = buffers[binding.buffer]?.descriptor else {
                throw GPUError.validation("Missing Metal runtime array binding")
            }
            let range = try binding.range(in: descriptor)
            let size = range.upperBound - range.lowerBound
            guard size >= UInt64(offset) + UInt64(stride),
                  let count = UInt32(exactly: (size - UInt64(offset)) / UInt64(stride)) else {
                throw GPUError.validation("Invalid Metal runtime array length")
            }
            return count
        }
    }

    func bindRender(
        _ encoder: any MTLRenderCommandEncoder, state: RenderState, pipeline: RenderPipeline,
        vertexBuffers: [VertexBufferBinding], uniforms: [ComputeBindingSlot: ComputeBufferBinding],
        storageBuffers: [ComputeBindingSlot: ComputeBufferBinding], vertexLengths: [UInt32], fragmentLengths: [UInt32],
        boundTextures: [ComputeBindingSlot: ResourceID], samplerBindings: [ComputeBindingSlot: ResourceID],
        blendConstant: RenderClearColor, raster: RenderRasterState
    ) {
        encoder.setRenderPipelineState(state.pipeline)
        if let viewport = raster.viewport {
            encoder.setViewport(MTLViewport(originX: Double(viewport.x), originY: Double(viewport.y),
                                            width: Double(viewport.width), height: Double(viewport.height),
                                            znear: Double(viewport.minDepth), zfar: Double(viewport.maxDepth)))
        }
        if let scissor = raster.scissor {
            encoder.setScissorRect(MTLScissorRect(x: Int(scissor.x), y: Int(scissor.y),
                                                 width: Int(scissor.width), height: Int(scissor.height)))
        }
        if pipeline.colorState.blend != nil || pipeline.additionalColorTargets.values.contains(where: { $0.colorState.blend != nil }) {
            encoder.setBlendColor(red: Float(blendConstant.red), green: Float(blendConstant.green),
                                  blue: Float(blendConstant.blue), alpha: Float(blendConstant.alpha))
        }
        encoder.setFrontFacing(pipeline.primitive.frontFace == .clockwise ? .clockwise : .counterClockwise)
        if capabilities.features.contains(.depthClipControl) {
            encoder.setDepthClipMode(pipeline.primitive.unclippedDepth ? .clamp : .clip)
        }
        switch pipeline.primitive.cullMode {
        case .none: encoder.setCullMode(.none)
        case .front: encoder.setCullMode(.front)
        case .back: encoder.setCullMode(.back)
        }
        if let depth = state.depth { encoder.setDepthStencilState(depth) }
        if let bias = pipeline.depthState?.bias, bias != .init() {
            encoder.setDepthBias(Float(bias.constant), slopeScale: bias.slopeScale, clamp: bias.clamp)
        }
        if state.writesDepth {
            var range = SIMD2<Float>(raster.viewport?.minDepth ?? 0, raster.viewport?.maxDepth ?? 1)
            encoder.setFragmentBytes(&range, length: MemoryLayout.size(ofValue: range),
                                     index: RenderStageArtifact.metalViewportDepthBufferIndex)
        }
        for (index, binding) in vertexBuffers.enumerated() {
            let slot = binding.slot.map(Int.init) ?? index
            guard slot < pipeline.vertexBuffers.count, !pipeline.vertexBuffers[slot].attributes.isEmpty else { continue }
            encoder.setVertexBuffer(buffers[binding.buffer]!.native, offset: Int(binding.offset), index: slot)
        }
        if state.usesVertexPulling, !pipeline.vertexBuffers.isEmpty {
            var sizes = [UInt64](repeating: 0, count: pipeline.vertexBuffers.count)
            for (index, binding) in vertexBuffers.enumerated() {
                let slot = binding.slot.map(Int.init) ?? index
                guard slot < sizes.count, !pipeline.vertexBuffers[slot].attributes.isEmpty else { continue }
                sizes[slot] = buffers[binding.buffer]!.descriptor.size - binding.offset
            }
            sizes.withUnsafeBytes {
                encoder.setVertexBytes($0.baseAddress!, length: $0.count,
                                       index: Int(RenderStageArtifact.metalVertexPullingSizeBufferIndex))
            }
        }
        for entry in pipeline.uniforms {
            let binding = uniforms[entry.slot]!
            let buffer = buffers[binding.buffer]!.native
            if let index = entry.vertexIndex {
                encoder.setVertexBuffer(buffer, offset: Int(binding.offset), index: Int(index))
            }
            if let index = entry.fragmentIndex {
                encoder.setFragmentBuffer(buffer, offset: Int(binding.offset), index: Int(index))
            }
        }
        for entry in pipeline.storageBuffers {
            let binding = storageBuffers[entry.handle.slot]!
            let buffer = buffers[binding.buffer]!.native
            if let index = entry.handle.vertexIndex {
                encoder.setVertexBuffer(buffer, offset: Int(binding.offset), index: Int(index))
            }
            if let index = entry.handle.fragmentIndex {
                encoder.setFragmentBuffer(buffer, offset: Int(binding.offset), index: Int(index))
            }
        }
        let lengthIndex = Int(RenderStageArtifact.metalStorageLengthBufferIndex)
        if !vertexLengths.isEmpty {
            vertexLengths.withUnsafeBytes { encoder.setVertexBytes($0.baseAddress!, length: $0.count, index: lengthIndex) }
        }
        if !fragmentLengths.isEmpty {
            fragmentLengths.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: lengthIndex) }
        }
        if let index = state.fragmentTextureAtomicGuardBufferIndex {
            var disabled: UInt32 = 0
            encoder.setFragmentBytes(&disabled, length: MemoryLayout<UInt32>.size, index: Int(index))
        }
        for entry in pipeline.sampledTextures.map(\.handle) + pipeline.storageTextures.map(\.handle) {
            let texture = nativeTexture(boundTextures[entry.slot]!)
            if let index = entry.vertexIndex { encoder.setVertexTexture(texture, index: Int(index)) }
            if let index = entry.fragmentIndex { encoder.setFragmentTexture(texture, index: Int(index)) }
        }
        bindExternalTextures(state.vertexExternalTextures, bindings: boundTextures,
                             setPlane: { encoder.setVertexTexture($0, index: $1) },
                             setParameters: { encoder.setVertexBytes($0, length: $1, index: $2) })
        bindExternalTextures(state.fragmentExternalTextures, bindings: boundTextures,
                             setPlane: { encoder.setFragmentTexture($0, index: $1) },
                             setParameters: { encoder.setFragmentBytes($0, length: $1, index: $2) })
        for binding in pipeline.samplerBindings {
            let entry = binding.handle
            let sampler = samplers[samplerBindings[entry.slot]!]!.native
            if let index = entry.vertexIndex { encoder.setVertexSamplerState(sampler, index: Int(index)) }
            if let index = entry.fragmentIndex { encoder.setFragmentSamplerState(sampler, index: Int(index)) }
        }
    }

    func fetchCapacity(
        _ layouts: [VertexBufferLayout], bindings: [VertexBufferBinding], stepMode: VertexStepMode
    ) -> UInt64 {
        var count = MetalIndirectDrawValidator.maximumCapacity
        for (index, binding) in bindings.enumerated() {
            let slot = binding.slot.map(Int.init) ?? index
            guard slot < layouts.count else { continue }
            let layout = layouts[slot]
            guard layout.stepMode == stepMode, !layout.attributes.isEmpty else { continue }
            let available = buffers[binding.buffer]!.descriptor.size - binding.offset
            guard available >= layout.attributeEnd else { return 0 }
            if layout.stride > 0 {
                count = min(count, (available - layout.attributeEnd) / UInt64(layout.stride) + 1)
            }
        }
        return count
    }

    func computeEncoder(
        _ commands: any MTLCommandBuffer, timestamps: PassTimestampWrites?
    ) throws -> any MTLComputeCommandEncoder {
        guard let timestamps else {
            guard let encoder = commands.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
            return encoder
        }
        let pass = MTLComputePassDescriptor()
        guard let attachment = pass.sampleBufferAttachments[0] else { throw GPUError.outOfMemory }
        attachment.sampleBuffer = querySets[timestamps.querySet]!.samples
        attachment.startOfEncoderSampleIndex = timestamps.beginningOfPassWriteIndex.map(Int.init) ?? MTLCounterDontSample
        attachment.endOfEncoderSampleIndex = timestamps.endOfPassWriteIndex.map(Int.init) ?? MTLCounterDontSample
        guard let encoder = commands.makeComputeCommandEncoder(descriptor: pass) else { throw GPUError.outOfMemory }
        return encoder
    }

    func clearSingleViewAttachments(
        _ commands: any MTLCommandBuffer, target: ResourceID?, level: UInt32, clear: MTLClearColor?,
        depth: RenderDepthAttachment?, additionalColors: [UInt32: RenderColorAttachment], viewCount: Int
    ) throws {
        let target = clear == nil ? nil : target
        let colors = additionalColors.compactMapValues { color -> RenderColorAttachment? in
            guard color.loadAction.clearColor != nil else { return nil }
            return .init(texture: color.texture, mipLevel: color.mipLevel, loadAction: color.loadAction)
        }
        var depthToClear: RenderDepthAttachment?
        if let depth {
            let depthLoad: RenderDepthAttachment.LoadAction = if case .clear = depth.loadAction { depth.loadAction } else { .unused }
            let stencil: RenderStencilAttachmentState? = if let stencil = depth.stencil, case .clear = stencil.loadAction {
                .init(loadAction: stencil.loadAction)
            } else {
                nil
            }
            if depthLoad != .unused || stencil != nil {
                depthToClear = .init(texture: depth.texture, mipLevel: depth.mipLevel, loadAction: depthLoad, stencil: stencil)
            }
        }
        guard target != nil || !colors.isEmpty || depthToClear != nil else { return }
        // An unamplified draw binds one slice. Load operations still cover the entire attachment view.
        try renderEncoder(commands, target: target.map { nativeTexture($0) }, level: level, clear: clear,
                          depth: depthToClear, additionalColors: colors, viewCount: viewCount).endEncoding()
    }

    func resolveLayeredAttachments(
        _ commands: any MTLCommandBuffer, target: ResourceID?, level: UInt32, resolve: RenderResolveAttachment?,
        colorStore: RenderStoreAction, additionalColors: [UInt32: RenderColorAttachment], viewCount: Int
    ) throws {
        let target = resolve == nil ? nil : target.map { nativeTexture($0) }
        let colors = additionalColors.compactMapValues { color -> RenderColorAttachment? in
            guard let resolve = color.resolve else { return nil }
            return .init(texture: color.texture, mipLevel: color.mipLevel, resolve: resolve, storeAction: color.storeAction)
        }
        // A single-view draw targets one slice, but resolves apply to the full attachment range.
        try renderEncoder(commands, target: target, level: level, clear: nil, resolve: resolve,
                          additionalColors: colors, colorStore: colorStore, viewCount: viewCount).endEncoding()
    }

    func clearDiscardedAttachments(
        _ commands: any MTLCommandBuffer, target: ResourceID?, level: UInt32, layer: UInt32,
        colorStore: RenderStoreAction, depth: RenderDepthAttachment?, additionalColors: [UInt32: RenderColorAttachment],
        viewCount: Int
    ) throws {
        let clear = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        let discardedTarget = colorStore == .discard ? target : nil
        let discardedColors = additionalColors.filter { $0.value.storeAction == .discard }.mapValues { color in
            RenderColorAttachment(texture: color.texture, mipLevel: color.mipLevel,
                                  loadAction: .clear(.transparentBlack), depthSlice: color.depthSlice)
        }
        var discardedDepth: RenderDepthAttachment?
        if let depth, depth.storeAction == .discard || depth.stencil?.storeAction == .discard {
            discardedDepth = .init(texture: depth.texture, mipLevel: depth.mipLevel,
                                   loadAction: depth.storeAction == .discard ? .clear(0) : .unused,
                                   stencil: depth.stencil?.storeAction == .discard ? .init() : nil)
        }
        guard discardedTarget != nil || !discardedColors.isEmpty || discardedDepth != nil else { return }
        // Metal leaves discarded contents undefined. Clear just those attachments after resolving.
        try renderEncoder(commands, target: discardedTarget.map { nativeTexture($0) }, level: level, layer: layer,
                          clear: clear, depth: discardedDepth, additionalColors: discardedColors, viewCount: viewCount).endEncoding()
    }

    func renderEncoder(
        _ commands: any MTLCommandBuffer, target: (any MTLTexture)?, level: UInt32 = 0, layer: UInt32 = 0, clear: MTLClearColor?,
        depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
        additionalColors: [UInt32: RenderColorAttachment] = [:], colorStore: RenderStoreAction = .store,
        timestamps: PassTimestampWrites? = nil,
        visibility: (any MTLBuffer)? = nil, viewCount: Int = 1, viewLayer: UInt32 = 0, deferResolve: Bool = false,
        rateMap: (any MTLRasterizationRateMap)? = nil
    ) throws -> any MTLRenderCommandEncoder {
        let pass = MTLRenderPassDescriptor()
        pass.visibilityResultBuffer = visibility
        pass.rasterizationRateMap = rateMap
        let attachmentLayers = rateMap?.layerCount ?? viewCount
        if attachmentLayers > 1 { pass.renderTargetArrayLength = attachmentLayers }
        if let timestamps {
            guard let attachment = pass.sampleBufferAttachments[0] else { throw GPUError.outOfMemory }
            attachment.sampleBuffer = querySets[timestamps.querySet]!.samples
            attachment.startOfVertexSampleIndex = timestamps.beginningOfPassWriteIndex.map(Int.init) ?? MTLCounterDontSample
            attachment.endOfVertexSampleIndex = MTLCounterDontSample
            attachment.startOfFragmentSampleIndex = MTLCounterDontSample
            attachment.endOfFragmentSampleIndex = timestamps.endOfPassWriteIndex.map(Int.init) ?? MTLCounterDontSample
        }
        func setColor(_ location: Int, texture: any MTLTexture, level: UInt32, layer: UInt32 = 0, clear: MTLClearColor?,
                      resolve: RenderResolveAttachment?, storeAction: RenderStoreAction) {
            let attachment = pass.colorAttachments[location]!
            attachment.texture = texture
            attachment.level = Int(level)
            if texture.textureType == .type3D {
                attachment.depthPlane = Int(layer)
            } else {
                attachment.slice = Int(layer)
            }
            attachment.loadAction = clear == nil ? .load : .clear
            attachment.storeAction = storeAction == .discard ? .dontCare : .store
            if let clear { attachment.clearColor = clear }
            if let resolve {
                if deferResolve {
                    attachment.storeAction = .store
                } else {
                    attachment.resolveTexture = nativeTexture(resolve.texture)
                    attachment.resolveLevel = Int(resolve.mipLevel)
                    attachment.storeAction = storeAction == .discard ? .multisampleResolve : .storeAndMultisampleResolve
                }
            }
        }
        if let target {
            setColor(0, texture: target, level: level, layer: layer + viewLayer, clear: clear, resolve: resolve, storeAction: colorStore)
        }
        for (location, attachment) in additionalColors {
            let clear = attachment.loadAction.clearColor.map {
                MTLClearColor(red: $0.red, green: $0.green, blue: $0.blue, alpha: $0.alpha)
            }
            setColor(Int(location), texture: nativeTexture(attachment.texture), level: attachment.mipLevel,
                     layer: attachment.depthSlice ?? viewLayer,
                     clear: clear, resolve: attachment.resolve, storeAction: attachment.storeAction)
        }
        if let depth {
            setDepthAttachment(pass, texture: nativeTexture(depth.texture), level: depth.mipLevel, layer: viewLayer,
                               loadAction: depth.loadAction, stencilLoadAction: depth.stencil?.loadAction,
                               storeAction: depth.storeAction, stencilStoreAction: depth.stencil?.storeAction ?? .store)
        }
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { throw GPUError.outOfMemory }
        return encoder
    }

    func setDepthAttachment(
        _ pass: MTLRenderPassDescriptor, texture: any MTLTexture, level: UInt32, layer: UInt32 = 0,
        loadAction: RenderDepthAttachment.LoadAction, stencilLoadAction: RenderStencilAttachmentState.LoadAction?,
        storeAction: RenderStoreAction = .store, stencilStoreAction: RenderStoreAction = .store
    ) {
        if loadAction != .unused {
            let attachment = pass.depthAttachment!
            attachment.texture = texture
            attachment.level = Int(level)
            attachment.slice = Int(layer)
            attachment.storeAction = storeAction == .discard ? .dontCare : .store
            if case .clear(let depth) = loadAction {
                attachment.loadAction = .clear
                attachment.clearDepth = depth
            } else {
                attachment.loadAction = .load
            }
        }
        if let stencilLoadAction {
            let stencil = pass.stencilAttachment!
            stencil.texture = texture
            stencil.level = Int(level)
            stencil.slice = Int(layer)
            stencil.storeAction = stencilStoreAction == .discard ? .dontCare : .store
            switch stencilLoadAction {
            case .load, .readOnly: stencil.loadAction = .load
            case .clear(let value):
                stencil.loadAction = .clear
                stencil.clearStencil = value & 0xff
            }
        }
    }
}
#endif
