import Foundation
import SAGABackend
import SAGAShaderIR
import SAGATypes
#if canImport(Metal)
import Metal

extension MetalDevice {
    /// A draw that can share a render encoder. Sampling and index validation are read-only.
    struct IndexedColorDraw {
        let pipeline: RenderPipeline
        let state: RenderState
        let target: ResourceID
        let level: UInt32
        let load: RenderColorLoadAction
        let additionalColors: [UInt32: RenderColorAttachment]
        let depth: RenderDepthAttachment?
        let index: IndexBufferBinding
        let draw: IndexedDraw
        let vertices: [VertexBufferBinding]
        let uniforms: [ComputeBindingSlot: ComputeBufferBinding]
        let storage: [ComputeBindingSlot: ComputeBufferBinding]
        let textures: [ComputeBindingSlot: ResourceID]
        let samplers: [ComputeBindingSlot: ResourceID]
        let blend: RenderClearColor
        let raster: RenderRasterState
        let maximumVertex: UInt32

        func follows(_ first: Self) -> Bool {
            guard target == first.target, level == first.level, load == .load,
                  additionalColors.count == first.additionalColors.count,
                  additionalColors.allSatisfy({ location, next in
                      guard let initial = first.additionalColors[location] else { return false }
                      return initial.texture == next.texture && initial.mipLevel == next.mipLevel
                          && initial.depthSlice == next.depthSlice && next.loadAction == .load
                  }) else { return false }
            switch (first.depth, depth) {
            case (nil, nil):
                return true
            case (let initial?, let next?):
                // A clear in any aspect starts a new encoder, even on the same allocation.
                return initial.texture == next.texture && initial.mipLevel == next.mipLevel
                    && next.loadAction == .load && next.stencil?.loadAction == .load
            default:
                return false
            }
        }
    }

    func indexedColorDraw(_ recorded: BufferCommand, index commandIndex: Int) -> IndexedColorDraw? {
        // Restrict this path to draws whose reordering of initialization and index checks is invisible.
        // Queries, copies, resolves and transitions through the same texture remain hard boundaries.
        guard case .renderIndexed(let pipeline, let optionalTarget, let load, let index, let draw, let vertices,
                                  let uniforms, let sampled, let samplers, let level, let layer, let depth,
                                  let resolve, let blend, let colors, let raster, let storage, let immediates, let store) = recorded,
              let target = optionalTarget, let color = textures[target], color.descriptor.sampleCount == 1,
              color.descriptor.dimension == .d2, color.descriptor.arrayLayerCount == 1,
              layer == nil || layer == 0,
              resolve == nil, store == .store,
              pipeline.multiviewMask == nil, pipeline.storageTextures.isEmpty,
              raster.rateMap == nil, !raster.hasEmptyScissor, immediates.isEmpty,
              let state = renderPipelines[pipeline.id], state.metalRenderTargetLayer == nil,
              canGroupIndexedAttachments(
                  color: color, additionalColors: colors, depth: depth, pipeline: pipeline, state: state,
                  eligibility: MetalIndexedDrawGrouping.commands[commandIndex]
              ),
              colors.isEmpty ? (pipeline.storageBuffers.isEmpty && storage.isEmpty)
                  : pipeline.storageBuffers.allSatisfy(\.isReadOnly),
              state.vertexExternalTextures.isEmpty, state.fragmentExternalTextures.isEmpty,
              state.fragmentTextureAtomicGuardBufferIndex == nil,
              state.vertexImmediateBufferIndex == nil, state.fragmentImmediateBufferIndex == nil,
              draw.indexCount > 0, draw.instances.count > 0,
              pipeline.vertexBuffers.contains(where: { $0.stepMode == .vertex && !$0.attributes.isEmpty }),
              sampled.values.allSatisfy({ id in
                  let parent = colors.isEmpty ? id : (textureViews[id]?.resource.texture ?? id)
                  guard let input = textures[parent] else { return false }
                  return input.native !== color.native && !input.descriptor.usage.contains(.storageBinding)
                      && depth.map { textures[$0.texture]?.native !== input.native } != false
                      && !colors.values.contains(where: { $0.texture == parent })
              })
        else { return nil }
        let capacity = fetchCapacity(pipeline.vertexBuffers, bindings: vertices, stepMode: .vertex)
        guard capacity > 0 else { return nil }
        return IndexedColorDraw(pipeline: pipeline, state: state, target: target, level: level, load: load,
                                additionalColors: colors, depth: depth, index: index, draw: draw, vertices: vertices,
                                uniforms: uniforms, storage: storage, textures: sampled, samplers: samplers, blend: blend, raster: raster,
                                maximumVertex: UInt32(capacity - 1))
    }

    private func canGroupIndexedAttachments(
        color: Texture, additionalColors: [UInt32: RenderColorAttachment], depth: RenderDepthAttachment?,
        pipeline: RenderPipeline, state: RenderState, eligibility: MetalIndexedDrawGrouping?
    ) -> Bool {
        if !additionalColors.isEmpty {
            guard eligibility == .geometry, depth != nil,
                  color.descriptor.format == .rgba16Float, pipeline.depthState?.writeEnabled == true,
                  additionalColors[1] != nil, additionalColors[2] != nil, additionalColors[3] != nil,
                  Set(pipeline.additionalColorTargets.keys) == Set(additionalColors.keys),
                  additionalColors.allSatisfy({ location, attachment in
                      guard (1...5).contains(location), let texture = textures[attachment.texture],
                            pipeline.additionalColorTargets[location]?.colorState.blend == nil else { return false }
                      let format = texture.descriptor.format
                      return (format == .rgba16Float || (location >= 4 && format == .rgba8Unorm))
                          && texture.descriptor.dimension == .d2 && texture.descriptor.arrayLayerCount == 1
                          && texture.descriptor.sampleCount == 1 && attachment.storeAction == .store
                          && attachment.resolve == nil && (attachment.depthSlice ?? 0) == 0
                  })
            else { return false }
        } else {
            guard pipeline.additionalColorTargets.isEmpty else { return false }
        }
        guard let depth else {
            return eligibility == .ui && pipeline.depthState == nil && state.depth == nil
        }
        let shadows = eligibility == .shadowDepth
            && (color.descriptor.format == .r32Float || color.descriptor.format == .rgba32Float)
        guard !additionalColors.isEmpty || shadows,
              let attachment = textures[depth.texture],
              attachment.descriptor.format == .depth24PlusStencil8 || attachment.descriptor.format == .depth32FloatStencil8,
              attachment.descriptor.sampleCount == 1,
              attachment.descriptor.dimension == .d2, attachment.descriptor.arrayLayerCount == 1,
              depth.storeAction == .store, depth.loadAction != .readOnly, depth.loadAction != .unused,
              let stencil = depth.stencil, stencil.storeAction == .store, stencil.loadAction != .readOnly,
              let depthState = pipeline.depthState, depthState.bias == .init(), depthState.stencil == nil,
              state.depth != nil
        else { return false }
        return true
    }

    func encodeIndexedColorGroup(
        _ draws: ArraySlice<IndexedColorDraw>, first: IndexedColorDraw, on command: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>],
        retained: inout [any MTLBuffer], pipelines: inout [RenderState]
    ) throws {
        for draw in draws {
            try initializeBoundTextures(draw.textures, on: command, initialized: &initialized)
        }
        try initializeLoadedAttachments(target: first.target, level: first.level, depthSlice: nil,
                                        colorLoad: first.load, depth: first.depth, additionalColors: first.additionalColors,
                                        on: command, initialized: &initialized)
        if indexValidator == nil { indexValidator = try MetalIndexValidator(device: native) }
        guard let validator = indexValidator else { throw GPUError.backendFailure("No Metal index validator") }
        let prepared = try validateIndexedColorGroup(draws, validator: validator, on: command)
        retained.append(contentsOf: prepared.map(\.arguments))

        let color = nativeTexture(first.target)
        let clear = first.load.clearColor.map { MTLClearColor(red: $0.red, green: $0.green, blue: $0.blue, alpha: $0.alpha) }
        let encoder = try renderEncoder(command, target: color, level: first.level, clear: clear,
                                        depth: first.depth, additionalColors: first.additionalColors)
        defer { encoder.endEncoding() }
        let width = max(1, color.width >> Int(first.level))
        let height = max(1, color.height >> Int(first.level))
        let fullViewport = MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1)
        let fullScissor = MTLScissorRect(x: 0, y: 0, width: width, height: height)
        for (draw, prepared) in zip(draws, prepared) {
            if draw.raster.viewport == nil { encoder.setViewport(fullViewport) }
            if draw.raster.scissor == nil { encoder.setScissorRect(fullScissor) }
            let vertexLengths = try runtimeLengthValues(draw.state.vertexRuntimeLengths, bindings: draw.storage)
            let fragmentLengths = try runtimeLengthValues(draw.state.fragmentRuntimeLengths, bindings: draw.storage)
            bindRender(encoder, state: draw.state, pipeline: draw.pipeline, vertexBuffers: draw.vertices,
                       uniforms: draw.uniforms, storageBuffers: draw.storage, vertexLengths: vertexLengths, fragmentLengths: fragmentLengths,
                       boundTextures: draw.textures, samplerBindings: draw.samplers, blendConstant: draw.blend, raster: draw.raster)
            if let stencil = draw.depth?.stencil { encoder.setStencilReferenceValue(stencil.reference & 0xff) }
            encoder.drawIndexedPrimitives(type: draw.pipeline.primitive.topology.metal.type, indexType: draw.index.format.metal,
                                          indexBuffer: prepared.indices, indexBufferOffset: prepared.offset,
                                          indirectBuffer: prepared.arguments, indirectBufferOffset: 0)
            pipelines.append(draw.state)
        }
        markRenderedAttachments(target: first.target, level: first.level, depthSlice: nil, depth: first.depth, resolve: nil,
                                additionalColors: first.additionalColors, initialized: &initialized)
    }

    private func validateIndexedColorGroup(
        _ draws: ArraySlice<IndexedColorDraw>, validator: MetalIndexValidator, on command: any MTLCommandBuffer
    ) throws -> [(indices: any MTLBuffer, offset: Int, arguments: any MTLBuffer)] {
        guard let encoder = command.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
        defer { encoder.endEncoding() }
        return try draws.map { draw in
            guard let indices = buffers[draw.index.buffer]?.native else {
                throw GPUError.validation("Metal draw group names an unknown index buffer")
            }
            let offset = Int(draw.index.offset + UInt64(draw.draw.firstIndex) * UInt64(draw.index.format.byteSize))
            let arguments = try validator.encode(on: encoder, device: native, indices: indices, format: draw.index.format,
                                                 byteOffset: offset, draw: draw.draw, maximumVertex: draw.maximumVertex,
                                                 primitiveRestart: draw.pipeline.primitive.topology.isStrip)
            return (indices, offset, arguments)
        }
    }
}
#endif
