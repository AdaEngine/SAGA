/// Initial queue commands. The diagnostic dispatch has a fixed two-buffer ABI.
public enum BufferCommand: Sendable, Equatable {
    case write(buffer: ResourceID, offset: UInt64, words: [UInt32])
    case dispatch(pipeline: ResourceID, input: ResourceID, output: ResourceID, count: UInt32)
    case dispatchCompute(pipeline: ComputePipeline, bindings: [ComputeBindingSlot: ResourceID],
                         workgroups: DispatchWorkgroups, textures: [ComputeBindingSlot: ResourceID] = [:],
                         samplers: [ComputeBindingSlot: ResourceID] = [:])
    case dispatchComputeRanges(pipeline: ComputePipeline, bindings: [ComputeBindingSlot: StorageBufferBinding],
                               workgroups: DispatchWorkgroups, textures: [ComputeBindingSlot: ResourceID] = [:],
                               samplers: [ComputeBindingSlot: ResourceID] = [:])
    case render(pipeline: RenderPipeline, target: ResourceID, colorLoad: RenderColorLoadAction,
                vertexCount: UInt32, vertexBuffers: [VertexBufferBinding] = [],
                firstVertex: UInt32 = 0, instances: DrawInstances = .init(),
                uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [:],
                textures: [ComputeBindingSlot: ResourceID] = [:],
                samplers: [ComputeBindingSlot: ResourceID] = [:], targetMipLevel: UInt32 = 0,
                depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
                blendConstant: RenderClearColor = .transparentBlack,
                additionalColors: [UInt32: RenderColorAttachment] = [:], raster: RenderRasterState = .init())
    case renderIndexed(pipeline: RenderPipeline, target: ResourceID, colorLoad: RenderColorLoadAction,
                       indexBuffer: IndexBufferBinding, draw: IndexedDraw, vertexBuffers: [VertexBufferBinding] = [],
                       uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [:],
                       textures: [ComputeBindingSlot: ResourceID] = [:],
                       samplers: [ComputeBindingSlot: ResourceID] = [:], targetMipLevel: UInt32 = 0,
                       depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
                       blendConstant: RenderClearColor = .transparentBlack,
                       additionalColors: [UInt32: RenderColorAttachment] = [:], raster: RenderRasterState = .init())
    case copyTextureToBuffer(source: ResourceID, destination: ResourceID, bytesPerRow: UInt32, mipLevel: UInt32 = 0,
                             aspect: TextureViewAspect = .all, destinationOffset: UInt64 = 0, arrayLayer: UInt32 = 0,
                             z: UInt32 = 0, depth: UInt32 = 1)
    case copyBufferToTexture(source: ResourceID, layout: TextureBufferLayout, destination: ResourceID,
                             region: TextureCopyRegion, aspect: TextureViewAspect = .all)
    case copyTextureToTexture(source: ResourceID, destination: ResourceID, sourceRegion: TextureCopyRegion,
                              destinationOrigin: TextureCopyOrigin = .init())
    case copy(source: ResourceID, destination: ResourceID, byteCount: UInt64)
    case read(buffer: ResourceID, byteCount: UInt64)

    public static func render(
        pipeline: RenderPipeline, target: ResourceID, clear: RenderClearColor, vertexCount: UInt32,
        vertexBuffers: [VertexBufferBinding] = [], firstVertex: UInt32 = 0, instances: DrawInstances = .init(),
        uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [:], textures: [ComputeBindingSlot: ResourceID] = [:],
        samplers: [ComputeBindingSlot: ResourceID] = [:], targetMipLevel: UInt32 = 0,
        depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
        blendConstant: RenderClearColor = .transparentBlack, additionalColors: [UInt32: RenderColorAttachment] = [:],
        raster: RenderRasterState = .init()
    ) -> Self {
        .render(pipeline: pipeline, target: target, colorLoad: .clear(clear), vertexCount: vertexCount,
                vertexBuffers: vertexBuffers, firstVertex: firstVertex, instances: instances, uniforms: uniforms,
                textures: textures, samplers: samplers, targetMipLevel: targetMipLevel,
                depth: depth, resolve: resolve, blendConstant: blendConstant, additionalColors: additionalColors, raster: raster)
    }

    public static func renderIndexed(
        pipeline: RenderPipeline, target: ResourceID, clear: RenderClearColor, indexBuffer: IndexBufferBinding,
        draw: IndexedDraw, vertexBuffers: [VertexBufferBinding] = [],
        uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [:], textures: [ComputeBindingSlot: ResourceID] = [:],
        samplers: [ComputeBindingSlot: ResourceID] = [:], targetMipLevel: UInt32 = 0,
        depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
        blendConstant: RenderClearColor = .transparentBlack, additionalColors: [UInt32: RenderColorAttachment] = [:],
        raster: RenderRasterState = .init()
    ) -> Self {
        .renderIndexed(pipeline: pipeline, target: target, colorLoad: .clear(clear), indexBuffer: indexBuffer, draw: draw,
                       vertexBuffers: vertexBuffers, uniforms: uniforms, textures: textures, samplers: samplers,
                       targetMipLevel: targetMipLevel, depth: depth, resolve: resolve, blendConstant: blendConstant, additionalColors: additionalColors, raster: raster)
    }

    public var resources: [ResourceID] {
        switch self {
        case .write(let id, _, _), .read(let id, _): [id]
        case .render(_, let target, _, _, let bindings, _, _, let uniforms, let textures, let samplers, _, let depth, let resolve, _, let additionalColors, _):
            [target] + bindings.map(\.buffer) + uniforms.values.map(\.buffer) + textures.values + samplers.values +
                [depth?.texture, resolve?.texture].compactMap { $0 } + additionalColors.values.flatMap { attachment in
                    [attachment.texture, attachment.resolve?.texture].compactMap { $0 }
                }
        case .renderIndexed(_, let target, _, let index, _, let bindings, let uniforms, let textures, let samplers, _, let depth, let resolve, _, let additionalColors, _):
            [target, index.buffer] + bindings.map(\.buffer) + uniforms.values.map(\.buffer) + textures.values + samplers.values +
                [depth?.texture, resolve?.texture].compactMap { $0 } + additionalColors.values.flatMap { attachment in
                    [attachment.texture, attachment.resolve?.texture].compactMap { $0 }
                }
        case .dispatch(_, let input, let output, _): [input, output]
        case .dispatchCompute(_, let bindings, _, let textures, let samplers):
            Array(bindings.values) + textures.values + samplers.values
        case .dispatchComputeRanges(_, let bindings, _, let textures, let samplers):
            bindings.values.map(\.buffer) + textures.values + samplers.values
        case .copy(let source, let destination, _), .copyTextureToBuffer(let source, let destination, _, _, _, _, _, _, _),
             .copyBufferToTexture(let source, _, let destination, _, _),
             .copyTextureToTexture(let source, let destination, _, _):
            [source, destination]
        }
    }

    package var computeDispatch: (pipeline: ComputePipeline, bindings: [ComputeBindingSlot: StorageBufferBinding],
                                  workgroups: DispatchWorkgroups, textures: [ComputeBindingSlot: ResourceID],
                                  samplers: [ComputeBindingSlot: ResourceID])? {
        switch self {
        case .dispatchCompute(let pipeline, let buffers, let groups, let textures, let samplers):
            (pipeline, buffers.mapValues { StorageBufferBinding(buffer: $0) }, groups, textures, samplers)
        case .dispatchComputeRanges(let pipeline, let bindings, let groups, let textures, let samplers):
            (pipeline, bindings, groups, textures, samplers)
        default: nil
        }
    }

    public func validate(
        device: UInt64, descriptors: [ResourceID: BufferDescriptor],
        textures: [ResourceID: TextureDescriptor] = [:], samplers: Set<ResourceID> = [],
        comparisonSamplers: Set<ResourceID> = []
    ) throws {
        let samplerDescriptors = Dictionary(uniqueKeysWithValues: samplers.union(comparisonSamplers).map {
            ($0, SamplerDescriptor(compare: comparisonSamplers.contains($0) ? .never : nil))
        })
        try validate(device: device, descriptors: descriptors, textures: textures,
                     samplerDescriptors: samplerDescriptors, textureViews: [:])
    }

    package func validate(
        device: UInt64, descriptors: [ResourceID: BufferDescriptor], textures: [ResourceID: TextureDescriptor],
        samplerDescriptors: [ResourceID: SamplerDescriptor], textureViews: [ResourceID: TextureViewResource],
        maximumViewportDimension: UInt32 = 8192
    ) throws {
        func require(_ id: ResourceID, _ usage: BufferUsage, _ size: UInt64, offset: UInt64 = 0) throws {
            guard id.device == device, let descriptor = descriptors[id], descriptor.usage.contains(usage),
                  size > 0, size % 4 == 0, offset % 4 == 0,
                  offset <= descriptor.size, size <= descriptor.size - offset else {
                throw GPUError.validation("Command has invalid resource, usage, size or alignment")
            }
        }
        func textureBinding(
            _ id: ResourceID?, usage: TextureUsage
        ) -> (view: TextureViewResource, texture: TextureDescriptor)? {
            guard let id, id.device == device else { return nil }
            let view = textureViews[id]
            let parent = view?.texture ?? id
            guard parent.device == device, let texture = textures[parent], texture.usage.contains(usage) else {
                return nil
            }
            return (view ?? .init(texture: id, levels: 0..<texture.mipLevelCount, layers: 0..<texture.arrayLayerCount,
                                  dimension: TextureViewDescriptor().resolvedDimension(in: texture)), texture)
        }
        func requireSampledTexture(_ id: ResourceID?, _ requirement: SampledTextureRequirement) throws -> TextureViewResource {
            guard let binding = textureBinding(id, usage: .textureBinding),
                  (binding.texture.sampleCount > 1) == requirement.isMultisampled,
                  binding.view.dimension == requirement.dimension else {
                throw GPUError.validation("Invalid sampled texture binding")
            }
            let sampleType: TextureSampleType?
            switch binding.view.aspect {
            case .all: sampleType = binding.texture.format.hasStencil ? nil : binding.texture.format.sampleType
            case .depthOnly: sampleType = .depth
            case .stencilOnly: sampleType = .uint
            }
            guard binding.texture.format.supports(aspect: binding.view.aspect), sampleType == requirement.sampleType else {
                throw GPUError.validation("Invalid sampled texture binding")
            }
            return binding.view
        }
        func requireSampler(_ id: ResourceID?, _ requirement: SamplerRequirement) throws {
            guard let id, id.device == device, let descriptor = samplerDescriptors[id],
                  (descriptor.compare != nil) == requirement.isComparison,
                  !requirement.requiresNonFiltering || (descriptor.minFilter == .nearest &&
                      descriptor.magFilter == .nearest && descriptor.mipmapFilter == .nearest) else {
                throw GPUError.validation("Invalid sampler binding")
            }
        }
        func requireStorageTextures(
            _ requirements: [StorageTextureRequirement], bindings: [ComputeBindingSlot: ResourceID],
            occupied: [(view: TextureViewResource, writable: Bool)], aliasError: String
        ) throws {
            var used = occupied
            for requirement in requirements {
                guard let binding = textureBinding(bindings[requirement.slot], usage: .storageBinding),
                      binding.texture.format == requirement.format, binding.view.levels.count == 1 else {
                    throw GPUError.validation("Storage texture binding requires one mip level of the declared format")
                }
                guard binding.view.dimension == (requirement.isArrayed ? .d2Array : .d2) else {
                    throw GPUError.validation("Storage texture view dimension does not match the shader binding")
                }
                let view = binding.view
                guard !used.contains(where: {
                    ($0.writable || requirement.writable) && $0.view.overlaps(view)
                }) else {
                    throw GPUError.validation(aliasError)
                }
                used.append((view, requirement.writable))
            }
        }
        func requireRender(
            _ pipeline: RenderPipeline, _ target: ResourceID, _ level: UInt32, _ colorLoad: RenderColorLoadAction,
            _ bindings: [VertexBufferBinding], uniforms: [ComputeBindingSlot: ComputeBufferBinding],
            boundTextures: [ComputeBindingSlot: ResourceID], samplerBindings: [ComputeBindingSlot: ResourceID],
            vertexEnd: UInt64, instanceEnd: UInt64, depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?,
            blendConstant: RenderClearColor, additionalColors: [UInt32: RenderColorAttachment], raster: RenderRasterState
        ) throws {
            guard pipeline.id.device == device, Set(additionalColors.keys) == Set(pipeline.additionalColorTargets.keys) else {
                throw GPUError.validation("Color attachments do not match the render pipeline")
            }
            guard blendConstant.channels.allSatisfy({ Float($0).isFinite }) else {
                throw GPUError.validation("Blend constant must contain finite float values")
            }
            guard (pipeline.depthState == nil) == (depth == nil) else {
                throw GPUError.validation("Depth attachment does not match the render pipeline")
            }

            var usedTextures: [(view: TextureViewResource, writable: Bool)] = []
            func addWritable(_ id: ResourceID, level: UInt32, layer: UInt32) throws {
                let view = TextureViewResource(texture: id, levels: level..<(level + 1), layers: layer..<(layer + 1))
                guard !usedTextures.contains(where: { $0.view.overlaps(view) }) else {
                    throw GPUError.validation("Aliased render attachments")
                }
                usedTextures.append((view, true))
            }
            func attachmentBinding(
                _ id: ResourceID, level: UInt32
            ) -> (id: ResourceID, level: UInt32, layer: UInt32, texture: TextureDescriptor)? {
                guard let binding = textureBinding(id, usage: .renderAttachment), binding.view.layers.count == 1 else {
                    return nil
                }
                if let view = textureViews[id] {
                    guard level == 0, view.levels.count == 1 else { return nil }
                    return (view.texture, view.levels.lowerBound, view.layers.lowerBound, binding.texture)
                }
                return (id, level, 0, binding.texture)
            }
            func addColor(_ attachment: RenderColorAttachment, format: TextureFormat) throws -> (width: UInt32, height: UInt32) {
                guard let binding = attachmentBinding(attachment.texture, level: attachment.mipLevel),
                      binding.texture.format == format, binding.texture.sampleCount == pipeline.sampleCount,
                      let extent = binding.texture.mipExtent(binding.level),
                      attachment.loadAction.clearColor?.isValid(for: format) ?? true else {
                    throw GPUError.validation("Invalid render pipeline, color attachment or clear value")
                }
                try addWritable(binding.id, level: binding.level, layer: binding.layer)
                if let resolve = attachment.resolve {
                    guard binding.texture.sampleCount > 1, format.isNormalizedColor,
                          let resolved = attachmentBinding(resolve.texture, level: resolve.mipLevel),
                          resolved.texture.sampleCount == 1, resolved.texture.format == format,
                          let size = resolved.texture.mipExtent(resolved.level), size == extent else {
                        throw GPUError.validation("Invalid color resolve attachment")
                    }
                    try addWritable(resolved.id, level: resolved.level, layer: resolved.layer)
                }
                return extent
            }
            let colorExtent = try addColor(.init(texture: target, mipLevel: level, loadAction: colorLoad, resolve: resolve),
                                           format: pipeline.colorFormat)
            try raster.validate(width: colorExtent.width, height: colorExtent.height, maximumViewportDimension: maximumViewportDimension)
            for (location, attachment) in additionalColors {
                guard let format = pipeline.additionalColorTargets[location]?.format,
                      try addColor(attachment, format: format) == colorExtent else {
                    throw GPUError.validation("Color attachment extents must match")
                }
            }
            if let depth, let state = pipeline.depthState {
                guard let binding = attachmentBinding(depth.texture, level: depth.mipLevel),
                      binding.texture.format == state.format, binding.texture.sampleCount == pipeline.sampleCount,
                      !binding.texture.format.hasStencil || textureViews[depth.texture] == nil || textureViews[depth.texture]?.aspect == .all,
                      let extent = binding.texture.mipExtent(binding.level), extent == colorExtent else {
                    throw GPUError.validation("Invalid depth attachment format, extent or subresource")
                }
                let descriptor = binding.texture
                guard descriptor.format.hasStencil == (depth.stencil != nil) else {
                    throw GPUError.validation("Stencil load action does not match the depth attachment format")
                }
                if case .clear(let value) = depth.loadAction, !(0...1).contains(value) {
                    throw GPUError.validation("Depth clear value must be between zero and one")
                }
                let depthReadOnly = depth.loadAction == .readOnly
                if depthReadOnly && state.writeEnabled {
                    throw GPUError.validation("Render pipeline writes to a read-only depth attachment")
                }
                let stencilReadOnly = depth.stencil?.loadAction == .readOnly
                if stencilReadOnly && state.stencil?.writes(cullMode: pipeline.primitive.cullMode) == true {
                    throw GPUError.validation("Render pipeline writes to a read-only stencil attachment")
                }
                let levels = binding.level..<(binding.level + 1)
                let layers = binding.layer..<(binding.layer + 1)
                usedTextures.append((.init(texture: binding.id, levels: levels, aspect: .depthOnly, layers: layers), !depthReadOnly))
                if depth.stencil != nil {
                    usedTextures.append((.init(texture: binding.id, levels: levels, aspect: .stencilOnly, layers: layers), !stencilReadOnly))
                }
            }
            guard bindings.count == pipeline.vertexBuffers.count else {
                throw GPUError.validation("Vertex buffers do not match pipeline slots")
            }
            guard uniforms.count == pipeline.uniforms.count else {
                throw GPUError.validation("Uniform bindings do not match the render pipeline layout")
            }
            guard boundTextures.count == pipeline.sampledTextures.count + pipeline.storageTextures.count else {
                throw GPUError.validation("Texture bindings do not match the render pipeline layout")
            }
            for texture in pipeline.sampledTextures {
                let requirement = texture.requirement
                let binding = try requireSampledTexture(boundTextures[requirement.slot], requirement)
                guard !usedTextures.contains(where: {
                    $0.writable && $0.view.overlaps(binding)
                }) else {
                    throw GPUError.validation("Render attachment cannot be bound as a sampled texture")
                }
                usedTextures.append((binding, false))
            }
            try requireStorageTextures(pipeline.storageTextures.map(\.requirement), bindings: boundTextures,
                                       occupied: usedTextures,
                                       aliasError: "Aliased writable render textures")
            guard samplerBindings.count == pipeline.samplerBindings.count else {
                throw GPUError.validation("Sampler bindings do not match the render pipeline layout")
            }
            for sampler in pipeline.samplerBindings {
                let requirement = sampler.requirement
                try requireSampler(samplerBindings[requirement.slot], requirement)
            }
            for requirement in pipeline.uniforms {
                guard let binding = uniforms[requirement.slot], binding.buffer.device == device,
                      let descriptor = descriptors[binding.buffer]
                else { throw GPUError.validation("Invalid render uniform binding") }
                let range = try binding.range(in: descriptor, usage: .uniform)
                guard range.upperBound - range.lowerBound >= requirement.minimumByteSize else {
                    throw GPUError.validation("Render uniform binding is smaller than the shader layout")
                }
            }
            let maximumEnd = UInt64(UInt32.max) + 1
            guard vertexEnd <= maximumEnd, instanceEnd <= maximumEnd else {
                throw GPUError.validation("Draw vertex or instance range exceeds 32-bit indices")
            }
            for (binding, layout) in zip(bindings, pipeline.vertexBuffers) {
                let end = layout.stepMode == .instance ? instanceEnd : vertexEnd
                let required = end == 0 ? 0 : (end - 1) * UInt64(layout.stride) + layout.attributeEnd
                guard binding.buffer.device == device, let buffer = descriptors[binding.buffer],
                      buffer.usage.contains(.vertex), binding.offset % 4 == 0,
                      binding.offset <= buffer.size, required <= buffer.size - binding.offset else {
                    throw GPUError.validation("Invalid vertex buffer, usage, offset or fetch range")
                }
            }
        }
        switch self {
        case .write(let id, let offset, let words):
            try require(id, .copyDestination, UInt64(words.count) * 4, offset: offset)
        case .dispatch(let pipeline, let input, let output, let count):
            guard pipeline.device == device, pipeline.serial > 0, input != output else {
                throw GPUError.validation("Invalid pipeline identity or aliased storage buffers")
            }
            try require(input, .storage, UInt64(count) * 4)
            try require(output, .storage, UInt64(count) * 4)
        case .dispatchCompute, .dispatchComputeRanges:
            let (pipeline, bindings, workgroups, boundTextures, samplerBindings) = computeDispatch!
            guard pipeline.id.device == device else { throw GPUError.validation("Foreign compute pipeline") }
            guard workgroups.x <= 65_535, workgroups.y <= 65_535, workgroups.z <= 65_535 else {
                throw GPUError.validation("Workgroup count exceeds the supported limit")
            }
            guard bindings.count == pipeline.bufferBindings.count,
                  boundTextures.count == pipeline.textureBindings.count + pipeline.storageTextureBindings.count,
                  samplerBindings.count == pipeline.samplerBindings.count else {
                throw GPUError.validation("Compute bindings do not match the pipeline layout")
            }
            let usedTextures = try pipeline.textureBindings.map {
                (try requireSampledTexture(boundTextures[$0.slot], $0), false)
            }
            try requireStorageTextures(pipeline.storageTextureBindings, bindings: boundTextures,
                                       occupied: usedTextures, aliasError: "Aliased writable compute textures")
            for requirement in pipeline.samplerBindings {
                try requireSampler(samplerBindings[requirement.slot], requirement)
            }
            var used: [ResourceID: [(range: Range<UInt64>, readOnly: Bool)]] = [:]
            for requirement in pipeline.bufferBindings {
                guard let binding = bindings[requirement.slot] else {
                    throw GPUError.validation("Compute bindings do not match the pipeline layout")
                }
                guard binding.buffer.device == device, let descriptor = descriptors[binding.buffer] else {
                    throw GPUError.validation("Command has invalid resource, usage, size or alignment")
                }
                let range = try binding.range(in: descriptor, usage: requirement.usage)
                guard range.upperBound - range.lowerBound >= requirement.minimumByteSize else {
                    throw GPUError.validation("Command has invalid resource, usage, size or alignment")
                }
                for previous in used[binding.buffer, default: []] {
                    if (!previous.readOnly || !requirement.isReadOnly) && previous.range.overlaps(range) {
                        throw GPUError.validation("Aliased writable compute buffers")
                    }
                    guard previous.readOnly == requirement.isReadOnly else {
                        throw GPUError.validation("Incompatible storage access modes for the same buffer")
                    }
                }
                used[binding.buffer, default: []].append((range, requirement.isReadOnly))
            }
        case .copy(let source, let destination, let size):
            guard source != destination else { throw GPUError.validation("Aliased copy") }
            try require(source, .copySource, size)
            try require(destination, .copyDestination, size)
        case .render(let pipeline, let target, let colorLoad, let count, let bindings, let firstVertex, let instances, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let depth, let resolve, let blendConstant, let additionalColors, let raster):
            try requireRender(pipeline, target, targetMipLevel, colorLoad, bindings, uniforms: uniforms, boundTextures: boundTextures,
                              samplerBindings: samplerBindings, vertexEnd: UInt64(firstVertex) + UInt64(count), instanceEnd: instances.end,
                              depth: depth, resolve: resolve, blendConstant: blendConstant, additionalColors: additionalColors, raster: raster)
        case .renderIndexed(let pipeline, let target, let colorLoad, let index, let draw, let bindings, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let depth, let resolve, let blendConstant, let additionalColors, let raster):
            try requireRender(pipeline, target, targetMipLevel, colorLoad, bindings, uniforms: uniforms, boundTextures: boundTextures,
                              samplerBindings: samplerBindings, vertexEnd: 0, instanceEnd: draw.instances.end,
                              depth: depth, resolve: resolve, blendConstant: blendConstant, additionalColors: additionalColors, raster: raster)
            guard !pipeline.primitive.topology.isStrip || pipeline.primitive.stripIndexFormat == index.format else {
                throw GPUError.validation("Indexed strips require a matching strip index format")
            }
            let stride = UInt64(index.format.byteSize)
            guard index.buffer.device == device, let buffer = descriptors[index.buffer],
                  buffer.usage.contains(.index), index.offset % stride == 0, index.offset <= buffer.size else {
                throw GPUError.validation("Invalid index buffer, usage or offset")
            }
            let remaining = buffer.size - index.offset
            let size = index.size ?? remaining
            let end = UInt64(draw.firstIndex) + UInt64(draw.indexCount)
            guard size <= remaining, end <= size / stride else {
                throw GPUError.validation("Indexed draw exceeds index buffer range")
            }
        case .copyTextureToBuffer(let source, let destination, let bytesPerRow, let mipLevel, let aspect, let offset,
                                  let layer, let z, let depth):
            guard source.device == device, let texture = textures[source], texture.usage.contains(.copySource),
                  texture.sampleCount == 1, let extent = texture.mipExtent(mipLevel), depth > 0,
                  TextureCopyRegion(width: extent.width, height: extent.height, mipLevel: mipLevel,
                                    arrayLayer: layer, z: z, depth: depth).fits(in: texture),
                  let texelBytes = texture.format.copyTexelBytes(aspect: aspect) else {
                throw GPUError.validation("Invalid texture copy source or aspect")
            }
            let size = try TextureBufferLayout(bytesPerRow: bytesPerRow)
                .byteCount(width: extent.width, height: extent.height, texelBytes: texelBytes, depth: depth)
            guard destination.device == device, let buffer = descriptors[destination], buffer.usage.contains(.copyDestination),
                  offset % UInt64(max(4, texelBytes)) == 0, offset <= buffer.size, size <= buffer.size - offset else {
                throw GPUError.validation("Invalid texture copy buffer, offset or size")
            }
        case .copyBufferToTexture(let source, let layout, let destination, let region, let aspect):
            guard destination.device == device, let texture = textures[destination],
                  texture.usage.contains(.copyDestination), texture.sampleCount == 1, region.fits(in: texture),
                  texture.format != .depth32Float, let texelBytes = texture.format.copyTexelBytes(aspect: aspect),
                  !texture.format.hasStencil || region.covers(texture) else {
                throw GPUError.validation("Invalid texture upload destination, region or aspect")
            }
            let count = try layout.byteCount(width: region.width, height: region.height, texelBytes: texelBytes, depth: region.depth)
            guard source.device == device, let buffer = descriptors[source], buffer.usage.contains(.copySource),
                  layout.offset % UInt64(max(4, texelBytes)) == 0,
                  layout.offset <= buffer.size, count <= buffer.size - layout.offset else {
                throw GPUError.validation("Invalid texture upload buffer, offset or size")
            }
        case .copyTextureToTexture(let source, let destination, let region, let origin):
            let destinationRegion = origin.region(width: region.width, height: region.height, depth: region.depth)
            guard source.device == device, destination.device == device,
                  let src = textures[source], let dst = textures[destination],
                  src.usage.contains(.copySource), dst.usage.contains(.copyDestination),
                  src.format == dst.format, src.sampleCount == dst.sampleCount, src.dimension == dst.dimension,
                  region.fits(in: src), destinationRegion.fits(in: dst) else {
                throw GPUError.validation("Invalid texture copy resources, formats, sample counts or regions")
            }
            guard source != destination || region.mipLevel != origin.mipLevel || region.arrayLayer != origin.arrayLayer else {
                throw GPUError.validation("Texture copy subresources must be disjoint")
            }
            if src.format.hasDepth || src.sampleCount > 1 {
                guard region.covers(src), destinationRegion.covers(dst) else {
                    throw GPUError.validation("Depth and multisampled texture copies require full mip extents")
                }
            }
        case .read(let id, let size):
            try require(id, .mapRead, size)
        }
    }
}
