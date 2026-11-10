/// Initial queue commands. The diagnostic dispatch has a fixed two-buffer ABI.
public enum BufferCommand: Sendable, Equatable {
    case buildTopLevelAccelerationStructure(destination: ResourceID, instances: [AccelerationStructureInstance])
    case buildAccelerationStructure(destination: ResourceID, geometry: AccelerationStructureTriangleGeometry)
    case write(buffer: ResourceID, offset: UInt64, words: [UInt32])
    case writeBytes(buffer: ResourceID, offset: UInt64, bytes: [UInt8], dataOffset: UInt64 = 0, byteCount: UInt64? = nil)
    case dispatch(pipeline: ResourceID, input: ResourceID, output: ResourceID, count: UInt32)
    case dispatchCompute(pipeline: ComputePipeline, bindings: [ComputeBindingSlot: ResourceID],
                         workgroups: DispatchWorkgroups, textures: [ComputeBindingSlot: ResourceID] = [:],
                         samplers: [ComputeBindingSlot: ResourceID] = [:], immediates: [UInt32] = [],
                         accelerationStructures: [ComputeBindingSlot: ResourceID] = [:])
    case dispatchComputeRanges(pipeline: ComputePipeline, bindings: [ComputeBindingSlot: StorageBufferBinding],
                               workgroups: DispatchWorkgroups, textures: [ComputeBindingSlot: ResourceID] = [:],
                               samplers: [ComputeBindingSlot: ResourceID] = [:], immediates: [UInt32] = [],
                         accelerationStructures: [ComputeBindingSlot: ResourceID] = [:])
    /// Reads three packed UInt32 workgroup counts (X, Y, Z) from the buffer at execution time.
    case dispatchComputeIndirect(pipeline: ComputePipeline, bindings: [ComputeBindingSlot: StorageBufferBinding],
                                 indirectBuffer: ResourceID, indirectOffset: UInt64 = 0,
                                 textures: [ComputeBindingSlot: ResourceID] = [:],
                                 samplers: [ComputeBindingSlot: ResourceID] = [:], immediates: [UInt32] = [],
                         accelerationStructures: [ComputeBindingSlot: ResourceID] = [:])
    case drawMesh(pipeline: MeshPipeline, target: ResourceID, colorLoad: RenderColorLoadAction, workgroups: DispatchWorkgroups,
                  taskBuffers: [ComputeBindingSlot: StorageBufferBinding] = [:], meshBuffers: [ComputeBindingSlot: StorageBufferBinding] = [:], taskTextures: [ComputeBindingSlot: ResourceID] = [:], meshTextures: [ComputeBindingSlot: ResourceID] = [:], taskSamplers: [ComputeBindingSlot: ResourceID] = [:], meshSamplers: [ComputeBindingSlot: ResourceID] = [:], additionalColors: [UInt32: RenderColorAttachment] = [:], colorStore: RenderStoreAction = .store, resolve: RenderResolveAttachment? = nil, depth: RenderDepthAttachment? = nil, indirectBuffer: ResourceID? = nil, indirectOffset: UInt64 = 0, fragmentBuffers: [ComputeBindingSlot: StorageBufferBinding] = [:], fragmentTextures: [ComputeBindingSlot: ResourceID] = [:], fragmentSamplers: [ComputeBindingSlot: ResourceID] = [:], raster: RenderRasterState = .init())
    case render(pipeline: RenderPipeline, target: ResourceID?, colorLoad: RenderColorLoadAction,
                vertexCount: UInt32, vertexBuffers: [VertexBufferBinding] = [],
                firstVertex: UInt32 = 0, instances: DrawInstances = .init(),
                uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [:],
                textures: [ComputeBindingSlot: ResourceID] = [:],
                samplers: [ComputeBindingSlot: ResourceID] = [:], targetMipLevel: UInt32 = 0, targetDepthSlice: UInt32? = nil,
                depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
                blendConstant: RenderClearColor = .transparentBlack,
                additionalColors: [UInt32: RenderColorAttachment] = [:], raster: RenderRasterState = .init(),
                storageBuffers: [ComputeBindingSlot: ComputeBufferBinding] = [:], immediates: [UInt32] = [],
                colorStore: RenderStoreAction = .store)
    case renderIndexed(pipeline: RenderPipeline, target: ResourceID?, colorLoad: RenderColorLoadAction,
                       indexBuffer: IndexBufferBinding, draw: IndexedDraw, vertexBuffers: [VertexBufferBinding] = [],
                       uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [:],
                       textures: [ComputeBindingSlot: ResourceID] = [:],
                       samplers: [ComputeBindingSlot: ResourceID] = [:], targetMipLevel: UInt32 = 0, targetDepthSlice: UInt32? = nil,
                       depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
                       blendConstant: RenderClearColor = .transparentBlack,
                       additionalColors: [UInt32: RenderColorAttachment] = [:], raster: RenderRasterState = .init(),
                       storageBuffers: [ComputeBindingSlot: ComputeBufferBinding] = [:], immediates: [UInt32] = [],
                colorStore: RenderStoreAction = .store)
    /// Reads packed draw arguments at execution time. Without an index buffer, the words are
    /// vertex count, instance count, first vertex and first instance. With an index buffer,
    /// they are index count, instance count, first index, signed base vertex and first instance.
    case renderIndirect(pipeline: RenderPipeline, target: ResourceID?, colorLoad: RenderColorLoadAction,
                        indirectBuffer: ResourceID, indirectOffset: UInt64 = 0, indexBuffer: IndexBufferBinding? = nil,
                        vertexBuffers: [VertexBufferBinding] = [],
                        uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [:],
                        textures: [ComputeBindingSlot: ResourceID] = [:],
                        samplers: [ComputeBindingSlot: ResourceID] = [:], targetMipLevel: UInt32 = 0,
                        targetDepthSlice: UInt32? = nil, depth: RenderDepthAttachment? = nil,
                        resolve: RenderResolveAttachment? = nil, blendConstant: RenderClearColor = .transparentBlack,
                        additionalColors: [UInt32: RenderColorAttachment] = [:], raster: RenderRasterState = .init(),
                        storageBuffers: [ComputeBindingSlot: ComputeBufferBinding] = [:], immediates: [UInt32] = [],
                colorStore: RenderStoreAction = .store)
    case copyTextureToBuffer(source: ResourceID, destination: ResourceID, bytesPerRow: UInt32, mipLevel: UInt32 = 0,
                             aspect: TextureViewAspect = .all, destinationOffset: UInt64 = 0, arrayLayer: UInt32 = 0,
                             z: UInt32 = 0, depth: UInt32 = 1, rowsPerImage: UInt32? = nil)
    /// Copies a texture region into a buffer, preserving bytes outside its footprint.
    case copyTextureRegionToBuffer(source: ResourceID, destination: ResourceID, layout: TextureBufferLayout,
                                   region: TextureCopyRegion, aspect: TextureViewAspect = .all)
    case copyBufferToTexture(source: ResourceID, layout: TextureBufferLayout, destination: ResourceID,
                             region: TextureCopyRegion, aspect: TextureViewAspect = .all)
    /// Uploads a region from CPU bytes. Depth counts Z slices or successive array layers.
    case writeTexture(destination: ResourceID, data: [UInt8], layout: TextureDataLayout = .init(),
                      region: TextureCopyRegion, aspect: TextureViewAspect = .all)
    case copyTextureToTexture(source: ResourceID, destination: ResourceID, sourceRegion: TextureCopyRegion,
                              destinationOrigin: TextureCopyOrigin = .init())
    /// Copies an aligned range. If omitted, the byte count extends to the end of the source.
    case copy(source: ResourceID, sourceOffset: UInt64 = 0, destination: ResourceID,
              destinationOffset: UInt64 = 0, byteCount: UInt64? = nil)
    case clearBuffer(buffer: ResourceID, offset: UInt64 = 0, byteCount: UInt64? = nil)
    case read(buffer: ResourceID, byteCount: UInt64)
    /// Attaches timestamp writes to a compute or render pass command.
    indirect case timestamped(pass: BufferCommand, writes: PassTimestampWrites)
    /// Measures visibility for one render pass. May be combined with timestamp writes on that pass.
    indirect case occlusionQueried(pass: BufferCommand, query: PassOcclusionQuery)
    case resolveQuerySet(querySet: ResourceID, firstQuery: UInt32, queryCount: UInt32,
                         destination: ResourceID, destinationOffset: UInt64 = 0)

    /// Draws using three packed UInt32 workgroup counts read from the GPU buffer at execution time.
    public static func drawMeshIndirect(
        pipeline: MeshPipeline, target: ResourceID, colorLoad: RenderColorLoadAction,
        indirectBuffer: ResourceID, indirectOffset: UInt64 = 0,
        taskBuffers: [ComputeBindingSlot: StorageBufferBinding] = [:], meshBuffers: [ComputeBindingSlot: StorageBufferBinding] = [:],
        taskTextures: [ComputeBindingSlot: ResourceID] = [:], meshTextures: [ComputeBindingSlot: ResourceID] = [:],
        taskSamplers: [ComputeBindingSlot: ResourceID] = [:], meshSamplers: [ComputeBindingSlot: ResourceID] = [:],
        additionalColors: [UInt32: RenderColorAttachment] = [:], colorStore: RenderStoreAction = .store,
        resolve: RenderResolveAttachment? = nil, depth: RenderDepthAttachment? = nil,
        fragmentBuffers: [ComputeBindingSlot: StorageBufferBinding] = [:], fragmentTextures: [ComputeBindingSlot: ResourceID] = [:], fragmentSamplers: [ComputeBindingSlot: ResourceID] = [:], raster: RenderRasterState = .init()
    ) -> Self {
        .drawMesh(pipeline: pipeline, target: target, colorLoad: colorLoad, workgroups: .init(x: 0),
                  taskBuffers: taskBuffers, meshBuffers: meshBuffers, taskTextures: taskTextures, meshTextures: meshTextures,
                  taskSamplers: taskSamplers, meshSamplers: meshSamplers, additionalColors: additionalColors,
                  colorStore: colorStore, resolve: resolve, depth: depth, indirectBuffer: indirectBuffer, indirectOffset: indirectOffset, fragmentBuffers: fragmentBuffers, fragmentTextures: fragmentTextures, fragmentSamplers: fragmentSamplers, raster: raster)
    }

    public static func render(
        pipeline: RenderPipeline, target: ResourceID, clear: RenderClearColor, vertexCount: UInt32,
        vertexBuffers: [VertexBufferBinding] = [], firstVertex: UInt32 = 0, instances: DrawInstances = .init(),
        uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [:], textures: [ComputeBindingSlot: ResourceID] = [:],
        samplers: [ComputeBindingSlot: ResourceID] = [:], targetMipLevel: UInt32 = 0, targetDepthSlice: UInt32? = nil,
        depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
        blendConstant: RenderClearColor = .transparentBlack, additionalColors: [UInt32: RenderColorAttachment] = [:],
        raster: RenderRasterState = .init(), storageBuffers: [ComputeBindingSlot: ComputeBufferBinding] = [:],
        immediates: [UInt32] = [], colorStore: RenderStoreAction = .store
    ) -> Self {
        .render(pipeline: pipeline, target: target, colorLoad: .clear(clear), vertexCount: vertexCount,
                vertexBuffers: vertexBuffers, firstVertex: firstVertex, instances: instances, uniforms: uniforms,
                textures: textures, samplers: samplers, targetMipLevel: targetMipLevel, targetDepthSlice: targetDepthSlice,
                depth: depth, resolve: resolve, blendConstant: blendConstant, additionalColors: additionalColors, raster: raster,
                storageBuffers: storageBuffers, immediates: immediates, colorStore: colorStore)
    }

    public static func renderIndexed(
        pipeline: RenderPipeline, target: ResourceID, clear: RenderClearColor, indexBuffer: IndexBufferBinding,
        draw: IndexedDraw, vertexBuffers: [VertexBufferBinding] = [],
        uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [:], textures: [ComputeBindingSlot: ResourceID] = [:],
        samplers: [ComputeBindingSlot: ResourceID] = [:], targetMipLevel: UInt32 = 0, targetDepthSlice: UInt32? = nil,
        depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
        blendConstant: RenderClearColor = .transparentBlack, additionalColors: [UInt32: RenderColorAttachment] = [:],
        raster: RenderRasterState = .init(), storageBuffers: [ComputeBindingSlot: ComputeBufferBinding] = [:],
        immediates: [UInt32] = [], colorStore: RenderStoreAction = .store
    ) -> Self {
        .renderIndexed(pipeline: pipeline, target: target, colorLoad: .clear(clear), indexBuffer: indexBuffer, draw: draw,
                       vertexBuffers: vertexBuffers, uniforms: uniforms, textures: textures, samplers: samplers,
                       targetMipLevel: targetMipLevel, targetDepthSlice: targetDepthSlice, depth: depth, resolve: resolve,
                       blendConstant: blendConstant, additionalColors: additionalColors, raster: raster,
                       storageBuffers: storageBuffers, immediates: immediates, colorStore: colorStore)
    }

    public var resources: [ResourceID] {
        switch self {
        case .buildAccelerationStructure(let destination, let geometry): [destination, geometry.vertexBuffer] + (geometry.indexBuffer.map { [$0.buffer] } ?? [])
        case .buildTopLevelAccelerationStructure(let destination, let instances): [destination] + instances.map(\.accelerationStructure)
        case .write(let id, _, _), .writeBytes(let id, _, _, _, _), .clearBuffer(let id, _, _), .read(let id, _): [id]
        case .drawMesh(let pipeline, let target, _, _, let taskBuffers, let meshBuffers, let taskTextures, let meshTextures, let taskSamplers, let meshSamplers, let additionalColors, _, let resolve, let depth, let indirect, _, let fragmentBuffers, let fragmentTextures, let fragmentSamplers, _):
            [pipeline.id, target] + (indirect.map { [$0] } ?? []) + [resolve?.texture, depth?.texture].compactMap { $0 } + fragmentBuffers.values.map(\.buffer) + Array(fragmentTextures.values) + Array(fragmentSamplers.values) + taskBuffers.values.map(\.buffer) + meshBuffers.values.map(\.buffer) + Array(taskTextures.values) + Array(meshTextures.values) + Array(taskSamplers.values) + Array(meshSamplers.values) + additionalColors.values.flatMap { [$0.texture, $0.resolve?.texture].compactMap { $0 } }
        case .render(_, let target, _, _, let bindings, _, _, let uniforms, let textures, let samplers, _, _, let depth, let resolve, _, let additionalColors, _, let storageBuffers, _, _):
            (target.map { [$0] } ?? []) + bindings.map(\.buffer) + uniforms.values.map(\.buffer) + storageBuffers.values.map(\.buffer) + textures.values + samplers.values +
                [depth?.texture, resolve?.texture].compactMap { $0 } + additionalColors.values.flatMap { attachment in
                    [attachment.texture, attachment.resolve?.texture].compactMap { $0 }
                }
        case .renderIndexed(_, let target, _, let index, _, let bindings, let uniforms, let textures, let samplers, _, _, let depth, let resolve, _, let additionalColors, _, let storageBuffers, _, _):
            (target.map { [$0] } ?? []) + [index.buffer] + bindings.map(\.buffer) + uniforms.values.map(\.buffer) + storageBuffers.values.map(\.buffer) + textures.values + samplers.values +
                [depth?.texture, resolve?.texture].compactMap { $0 } + additionalColors.values.flatMap { attachment in
                    [attachment.texture, attachment.resolve?.texture].compactMap { $0 }
                }
        case .renderIndirect(_, let target, _, let arguments, _, let index, let bindings, let uniforms, let textures, let samplers, _, _, let depth, let resolve, _, let additionalColors, _, let storageBuffers, _, _):
            (target.map { [$0] } ?? []) + [arguments] + bindings.map(\.buffer) + uniforms.values.map(\.buffer) + storageBuffers.values.map(\.buffer) + textures.values + samplers.values +
                [index?.buffer, depth?.texture, resolve?.texture].compactMap { $0 } + additionalColors.values.flatMap { attachment in
                    [attachment.texture, attachment.resolve?.texture].compactMap { $0 }
                }
        case .dispatch(_, let input, let output, _): [input, output]
        case .dispatchCompute(_, let bindings, _, let textures, let samplers, _, let structures):
            Array(bindings.values) + textures.values + samplers.values + structures.values
        case .dispatchComputeRanges(_, let bindings, _, let textures, let samplers, _, let structures):
            bindings.values.map(\.buffer) + textures.values + samplers.values + structures.values
        case .dispatchComputeIndirect(_, let bindings, let arguments, _, let textures, let samplers, _, let structures):
            [arguments] + bindings.values.map(\.buffer) + textures.values + samplers.values + structures.values
        case .timestamped, .occlusionQueried:
            measuredPassResources
        case .resolveQuerySet(let set, _, _, let destination, _):
            [set, destination]
        case .writeTexture(let destination, _, _, _, _): [destination]
        case .copy(let source, _, let destination, _, _), .copyTextureToBuffer(let source, let destination, _, _, _, _, _, _, _, _),
             .copyTextureRegionToBuffer(let source, let destination, _, _, _),
             .copyBufferToTexture(let source, _, let destination, _, _),
             .copyTextureToTexture(let source, let destination, _, _):
            [source, destination]
        }
    }

    package static func uploadBytes(_ bytes: [UInt8], dataOffset: UInt64, byteCount: UInt64?) throws -> ArraySlice<UInt8> {
        let available = UInt64(bytes.count)
        guard dataOffset <= available else { throw GPUError.validation("Upload source offset exceeds the byte array") }
        let count = byteCount ?? (available - dataOffset)
        guard count <= available - dataOffset else { throw GPUError.validation("Upload source range exceeds the byte array") }
        return bytes[Int(dataOffset)..<Int(dataOffset + count)]
    }

    private var measuredPassResources: [ResourceID] {
        let (pass, timestamps, occlusion) = recordedPass
        let sets = [timestamps?.querySet, occlusion?.querySet].compactMap { $0 }
        return sets + (pass.acceptsTimestampWrites ? pass.resources : [])
    }

    private var acceptsTimestampWrites: Bool {
        switch self {
        case .dispatchCompute, .dispatchComputeRanges, .dispatchComputeIndirect, .render, .renderIndexed, .renderIndirect, .drawMesh: true
        default: false
        }
    }

    private var isRenderPass: Bool {
        switch self {
        case .render, .renderIndexed, .renderIndirect, .drawMesh: true
        default: false
        }
    }

    package var recordedPass: (command: BufferCommand, timestamps: PassTimestampWrites?, occlusion: PassOcclusionQuery?) {
        var command = self
        var timestamps: PassTimestampWrites?
        var occlusion: PassOcclusionQuery?
        for _ in 0..<2 {
            switch command {
            case .timestamped(let pass, let writes) where timestamps == nil:
                command = pass
                timestamps = writes
            case .occlusionQueried(let pass, let query) where occlusion == nil:
                command = pass
                occlusion = query
            default:
                return (command, timestamps, occlusion)
            }
        }
        return (command, timestamps, occlusion)
    }

    package enum DispatchSource {
        case direct(DispatchWorkgroups)
        case indirect(buffer: ResourceID, offset: UInt64)
    }

    package var computeDispatch: (pipeline: ComputePipeline, bindings: [ComputeBindingSlot: StorageBufferBinding],
                                  source: DispatchSource, textures: [ComputeBindingSlot: ResourceID],
                                  samplers: [ComputeBindingSlot: ResourceID], immediates: [UInt32],
                                  accelerationStructures: [ComputeBindingSlot: ResourceID])? {
        switch self {
        case .dispatchCompute(let pipeline, let buffers, let groups, let textures, let samplers, let immediates, let structures):
            (pipeline, buffers.mapValues { StorageBufferBinding(buffer: $0) }, .direct(groups), textures, samplers, immediates, structures)
        case .dispatchComputeRanges(let pipeline, let bindings, let groups, let textures, let samplers, let immediates, let structures):
            (pipeline, bindings, .direct(groups), textures, samplers, immediates, structures)
        case .dispatchComputeIndirect(let pipeline, let bindings, let buffer, let offset, let textures, let samplers, let immediates, let structures):
            (pipeline, bindings, .indirect(buffer: buffer, offset: offset), textures, samplers, immediates, structures)
        default: nil
        }
    }

    public func validate(
        device: UInt64, descriptors: [ResourceID: BufferDescriptor],
        textures: [ResourceID: TextureDescriptor] = [:], samplers: Set<ResourceID> = [],
        comparisonSamplers: Set<ResourceID> = [], querySets: [ResourceID: QuerySetDescriptor] = [:],
        features: Set<GPUFeature> = [], maxComputeWorkgroupsPerDimension: UInt32 = 65_535,
        accelerationStructures: [ResourceID: AccelerationStructureDescriptor] = [:],
        maxTaskWorkgroupTotalCount: UInt32 = .max, maxTaskWorkgroupsPerDimension: UInt32 = .max,
        maxMeshWorkgroupTotalCount: UInt32 = .max, maxMeshWorkgroupsPerDimension: UInt32 = .max
    ) throws {
        let samplerDescriptors = Dictionary(uniqueKeysWithValues: samplers.union(comparisonSamplers).map {
            ($0, SamplerDescriptor(compare: comparisonSamplers.contains($0) ? .never : nil))
        })
        try validate(device: device, descriptors: descriptors, textures: textures,
                     samplerDescriptors: samplerDescriptors, textureViews: [:], querySets: querySets, features: features,
                     accelerationStructures: accelerationStructures, maxComputeWorkgroupsPerDimension: maxComputeWorkgroupsPerDimension,
                     maxTaskWorkgroupTotalCount: maxTaskWorkgroupTotalCount, maxTaskWorkgroupsPerDimension: maxTaskWorkgroupsPerDimension,
                     maxMeshWorkgroupTotalCount: maxMeshWorkgroupTotalCount, maxMeshWorkgroupsPerDimension: maxMeshWorkgroupsPerDimension)
    }

    package func validate(
        device: UInt64, descriptors: [ResourceID: BufferDescriptor], textures: [ResourceID: TextureDescriptor],
        samplerDescriptors: [ResourceID: SamplerDescriptor], textureViews: [ResourceID: TextureViewResource],
        maximumViewportDimension: UInt32 = 8192, maxVertexBuffers: UInt32 = 8,
        maxBindGroupsPlusVertexBuffers: UInt32 = 24, querySets: [ResourceID: QuerySetDescriptor] = [:],
        maxMultiviewViewCount: UInt32 = 0,
        features: Set<GPUFeature> = [], externalTextures: Set<ResourceID> = [],
        externalPlanes: [ResourceID: [ResourceID]] = [:],
        accelerationStructures: [ResourceID: AccelerationStructureDescriptor] = [:],
        maxComputeWorkgroupsPerDimension: UInt32 = DispatchWorkgroups.defaultMaximumDimension,
        maxTaskWorkgroupTotalCount: UInt32 = .max, maxTaskWorkgroupsPerDimension: UInt32 = .max,
        maxMeshWorkgroupTotalCount: UInt32 = .max, maxMeshWorkgroupsPerDimension: UInt32 = .max
    ) throws {
        @discardableResult
        func require(
            _ id: ResourceID, _ usage: BufferUsage, _ size: UInt64?, offset: UInt64 = 0, allowEmpty: Bool = false
        ) throws -> UInt64 {
            guard id.device == device, let descriptor = descriptors[id], descriptor.usage.contains(usage),
                  offset % 4 == 0, offset <= descriptor.size else {
                throw GPUError.validation("Command has invalid resource, usage, size or alignment")
            }
            let count = size ?? (descriptor.size - offset)
            guard (allowEmpty || count > 0), count % 4 == 0, count <= descriptor.size - offset else {
                throw GPUError.validation("Command has invalid resource, usage, size or alignment")
            }
            return count
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
            case .all: sampleType = binding.texture.format.hasStencil && binding.texture.format.hasDepth ? nil : binding.texture.format.sampleType
            case .depthOnly: sampleType = .depth
            case .stencilOnly: sampleType = .uint
            }
            guard binding.texture.format.supports(aspect: binding.view.aspect), sampleType == requirement.sampleType else {
                throw GPUError.validation("Invalid sampled texture binding")
            }
            if requirement.requiresFiltering, binding.texture.format.isFloat32Color, !features.contains(.float32Filterable) {
                throw GPUError.validation("Float32 sampling requires float32-filterable")
            }
            if requirement.requiresFiltering, binding.texture.format.isNormalized16Color, !features.contains(.normalized16Filterable) {
                throw GPUError.validation("Normalized 16-bit sampling requires normalized16Filterable")
            }
            return binding.view
        }
        func requireExternalTexture(_ id: ResourceID?) throws -> [TextureViewResource] {
            guard let id, id.device == device, externalTextures.contains(id) else {
                throw GPUError.validation("Invalid external texture binding")
            }
            return try (externalPlanes[id] ?? []).map { plane in
                guard let view = textureBinding(plane, usage: .textureBinding)?.view else {
                    throw GPUError.validation("Invalid external texture plane")
                }
                let level = view.levels.lowerBound
                return .init(texture: view.texture, levels: level..<(level + 1), aspect: view.aspect,
                             layers: view.layers, dimension: view.dimension)
            }
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
                let usage: TextureUsage = requirement.access == .atomic ? .storageAtomic : .storageBinding
                guard let binding = textureBinding(bindings[requirement.slot], usage: usage),
                      binding.texture.format == requirement.format, binding.view.levels.count == 1 else {
                    throw GPUError.validation("Storage texture binding requires one mip level of the declared format")
                }
                guard binding.view.dimension == requirement.dimension else {
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
            _ pipeline: RenderPipeline, _ target: ResourceID?, _ level: UInt32, _ colorLoad: RenderColorLoadAction,
            targetDepthSlice: UInt32?, colorStore: RenderStoreAction,
            _ bindings: [VertexBufferBinding], uniforms: [ComputeBindingSlot: ComputeBufferBinding],
            storageBuffers: [ComputeBindingSlot: ComputeBufferBinding], indexBuffer: ResourceID? = nil,
            indirectBuffer: ResourceID? = nil,
            boundTextures: [ComputeBindingSlot: ResourceID], samplerBindings: [ComputeBindingSlot: ResourceID],
            vertexEnd: UInt64, instanceEnd: UInt64, depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?,
            blendConstant: RenderClearColor, additionalColors: [UInt32: RenderColorAttachment], raster: RenderRasterState,
            immediates: [UInt32]
        ) throws {
            guard pipeline.id.device == device, (target == nil) == (pipeline.colorFormat == nil),
                  Set(additionalColors.keys) == Set(pipeline.additionalColorTargets.keys) else {
                throw GPUError.validation("Color attachments do not match the render pipeline")
            }
            guard UInt64(immediates.count) == pipeline.immediateByteSize / 4 else {
                throw GPUError.validation("Render immediate data size does not match the pipeline layout")
            }
            guard blendConstant.channels.allSatisfy({ Float($0).isFinite }) else {
                throw GPUError.validation("Blend constant must contain finite float values")
            }
            guard (pipeline.depthState == nil) == (depth == nil) else {
                throw GPUError.validation("Depth attachment does not match the render pipeline")
            }

            var usedTextures: [(view: TextureViewResource, writable: Bool)] = []
            var colorRegions: [(view: TextureViewResource, depthSlice: UInt32?)] = []
            var renderLayerCount: UInt32?
            func addWritable(_ id: ResourceID, level: UInt32, layers: Range<UInt32>, depthSlice: UInt32? = nil) throws {
                let view = TextureViewResource(texture: id, levels: level..<(level + 1), layers: layers)
                guard !colorRegions.contains(where: { $0.view.overlaps(view) && $0.depthSlice == depthSlice }) else {
                    throw GPUError.validation("Aliased render attachments")
                }
                colorRegions.append((view, depthSlice))
                // Shader bindings cover every depth slice of a 3D mip.
                usedTextures.append((view, true))
            }
            func attachmentBinding(
                _ id: ResourceID, level: UInt32, depthSlice: UInt32? = nil
            ) -> (id: ResourceID, level: UInt32, layers: Range<UInt32>, texture: TextureDescriptor)? {
                guard let binding = textureBinding(id, usage: .renderAttachment) else { return nil }
                let layers = binding.view.layers
                if let mask = pipeline.multiviewMask {
                    let count = UInt32(layers.count)
                    let span = 32 - mask.leadingZeroBitCount
                    guard features.contains(.multiview), span > 0, span <= count, count <= maxMultiviewViewCount,
                          binding.view.dimension == .d2Array || (count == 1 && binding.view.dimension == .d2),
                          renderLayerCount.map({ $0 == count }) ?? true,
                          mask.nonzeroBitCount == count || features.contains(.selectiveMultiview) else { return nil }
                    renderLayerCount = count
                } else if layers.count != 1 {
                    return nil
                }
                let view = textureViews[id]
                if let view, level != 0 || view.levels.count != 1 { return nil }
                let mip = view?.levels.lowerBound ?? level
                if binding.texture.dimension == .d3 {
                    guard let depthSlice, depthSlice < binding.texture.mipDepth(mip) else { return nil }
                } else if depthSlice != nil {
                    return nil
                }
                return (view?.texture ?? id, mip, layers, binding.texture)
            }
            func addColor(_ attachment: RenderColorAttachment, format: TextureFormat) throws -> (width: UInt32, height: UInt32) {
                guard let binding = attachmentBinding(attachment.texture, level: attachment.mipLevel,
                                                      depthSlice: attachment.depthSlice),
                      binding.texture.format == format, binding.texture.sampleCount == pipeline.sampleCount,
                      let extent = binding.texture.mipExtent(binding.level),
                      attachment.loadAction.clearColor?.isValid(for: format) ?? true else {
                    throw GPUError.validation("Invalid render pipeline, color attachment or clear value")
                }
                try addWritable(binding.id, level: binding.level, layers: binding.layers, depthSlice: attachment.depthSlice)
                if let resolve = attachment.resolve {
                    guard binding.texture.sampleCount > 1, format.supportsColorResolve(features: features),
                          let resolved = attachmentBinding(resolve.texture, level: resolve.mipLevel),
                          resolved.texture.sampleCount == 1, resolved.texture.format == format,
                          let size = resolved.texture.mipExtent(resolved.level), size == extent else {
                        throw GPUError.validation("Invalid color resolve attachment")
                    }
                    try addWritable(resolved.id, level: resolved.level, layers: resolved.layers)
                }
                return extent
            }
            var renderExtent: (width: UInt32, height: UInt32)?
            if let target, let format = pipeline.colorFormat {
                renderExtent = try addColor(.init(texture: target, mipLevel: level, loadAction: colorLoad,
                                                 resolve: resolve, depthSlice: targetDepthSlice, storeAction: colorStore), format: format)
            } else if level != 0 || targetDepthSlice != nil || resolve != nil || colorLoad != .load || colorStore != .store {
                throw GPUError.validation("Color attachment state requires a primary target")
            }
            for (location, attachment) in additionalColors {
                guard let format = pipeline.additionalColorTargets[location]?.format else {
                    throw GPUError.validation("Color attachments do not match the render pipeline")
                }
                let extent = try addColor(attachment, format: format)
                guard renderExtent.map({ $0 == extent }) ?? true else {
                    throw GPUError.validation("Color attachment extents must match")
                }
                renderExtent = extent
            }
            if let depth, let state = pipeline.depthState {
                guard let binding = attachmentBinding(depth.texture, level: depth.mipLevel),
                      binding.texture.format == state.format, binding.texture.sampleCount == pipeline.sampleCount,
                      !binding.texture.format.hasStencil || !binding.texture.format.hasDepth ||
                          textureViews[depth.texture] == nil || textureViews[depth.texture]?.aspect == .all,
                      let extent = binding.texture.mipExtent(binding.level),
                      renderExtent.map({ $0 == extent }) ?? true else {
                    throw GPUError.validation("Invalid depth attachment format, extent or subresource")
                }
                renderExtent = extent
                let descriptor = binding.texture
                guard descriptor.format.hasStencil == (depth.stencil != nil) else {
                    throw GPUError.validation("Stencil load action does not match the depth attachment format")
                }
                guard descriptor.format.hasDepth == (depth.loadAction != .unused) else {
                    throw GPUError.validation("Depth load action does not match the attachment format")
                }
                if case .clear(let value) = depth.loadAction, !(0...1).contains(value) {
                    throw GPUError.validation("Depth clear value must be between zero and one")
                }
                let depthReadOnly = depth.loadAction == .readOnly
                if depth.storeAction == .discard && (depthReadOnly || depth.loadAction == .unused) ||
                    (depth.stencil?.storeAction == .discard && depth.stencil?.loadAction == .readOnly) {
                    throw GPUError.validation("Read-only or unused aspects cannot be discarded")
                }
                if depthReadOnly && state.writeEnabled {
                    throw GPUError.validation("Render pipeline writes to a read-only depth attachment")
                }
                let stencilReadOnly = depth.stencil?.loadAction == .readOnly
                if stencilReadOnly && state.stencil?.writes(cullMode: pipeline.primitive.cullMode) == true {
                    throw GPUError.validation("Render pipeline writes to a read-only stencil attachment")
                }
                let levels = binding.level..<(binding.level + 1)
                let layers = binding.layers
                if descriptor.format.hasDepth {
                    usedTextures.append((.init(texture: binding.id, levels: levels, aspect: .depthOnly, layers: layers), !depthReadOnly))
                }
                if depth.stencil != nil {
                    usedTextures.append((.init(texture: binding.id, levels: levels, aspect: .stencilOnly, layers: layers), !stencilReadOnly))
                }
            }
            guard let renderExtent else { throw GPUError.validation("Render pass requires at least one attachment") }
            try raster.validate(width: renderExtent.width, height: renderExtent.height, maximumViewportDimension: maximumViewportDimension)
            var vertexBindings: [Int: VertexBufferBinding] = [:]
            for (index, binding) in bindings.enumerated() {
                guard let slot = binding.slot.map({ Int(exactly: $0) }) ?? index,
                      UInt64(slot) < UInt64(maxVertexBuffers), vertexBindings.updateValue(binding, forKey: slot) == nil else {
                    throw GPUError.validation("Invalid or duplicate vertex buffer slot")
                }
            }
            let groups = uniforms.keys.map(\.group) + storageBuffers.keys.map(\.group)
                + boundTextures.keys.map(\.group) + samplerBindings.keys.map(\.group)
            let groupSpan = groups.max().map { UInt64($0) + 1 } ?? 0
            let vertexSpan = vertexBindings.keys.max().map { UInt64($0) + 1 } ?? 0
            guard groupSpan + vertexSpan <= UInt64(maxBindGroupsPlusVertexBuffers) else {
                throw GPUError.validation("Bind groups and vertex buffer slots exceed the combined limit")
            }
            guard pipeline.vertexBuffers.enumerated().allSatisfy({ $0.element.attributes.isEmpty || vertexBindings[$0.offset] != nil }) else {
                throw GPUError.validation("Vertex buffers do not match pipeline slots")
            }
            guard uniforms.count == pipeline.uniforms.count else {
                throw GPUError.validation("Uniform bindings do not match the render pipeline layout")
            }
            guard storageBuffers.count == pipeline.storageBuffers.count else {
                throw GPUError.validation("Storage buffer bindings do not match the render pipeline layout")
            }
            guard boundTextures.count == pipeline.sampledTextures.count + pipeline.storageTextures.count + pipeline.externalTextureBindings.count else {
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
            for external in pipeline.externalTextureBindings {
                for plane in try requireExternalTexture(boundTextures[external.slot]) {
                    guard !usedTextures.contains(where: { $0.writable && $0.view.overlaps(plane) }) else {
                        throw GPUError.validation("Render attachment cannot be bound as an external texture plane")
                    }
                    usedTextures.append((plane, false))
                }
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
            var readOnlyBuffers = Set(uniforms.values.map(\.buffer) + bindings.map(\.buffer))
            if let indexBuffer { readOnlyBuffers.insert(indexBuffer) }
            if let indirectBuffer { readOnlyBuffers.insert(indirectBuffer) }
            var writableRanges: [ResourceID: [Range<UInt64>]] = [:]
            for requirement in pipeline.storageBuffers {
                guard let binding = storageBuffers[requirement.handle.slot], binding.buffer.device == device,
                      let descriptor = descriptors[binding.buffer] else {
                    throw GPUError.validation("Invalid render storage buffer binding")
                }
                let range = try binding.range(in: descriptor, usage: .storage)
                guard range.upperBound - range.lowerBound >= requirement.minimumByteSize else {
                    throw GPUError.validation("Render storage binding is smaller than the shader layout")
                }
                if requirement.isReadOnly {
                    guard writableRanges[binding.buffer] == nil else {
                        throw GPUError.validation("Aliased writable render buffers")
                    }
                    readOnlyBuffers.insert(binding.buffer)
                } else {
                    guard !readOnlyBuffers.contains(binding.buffer),
                          !writableRanges[binding.buffer, default: []].contains(where: { $0.overlaps(range) }) else {
                        throw GPUError.validation("Aliased writable render buffers")
                    }
                    writableRanges[binding.buffer, default: []].append(range)
                }
            }
            let maximumEnd = UInt64(UInt32.max) + 1
            guard vertexEnd <= maximumEnd, instanceEnd <= maximumEnd else {
                throw GPUError.validation("Draw vertex or instance range exceeds 32-bit indices")
            }
            for (slot, binding) in vertexBindings {
                let layout = slot < pipeline.vertexBuffers.count ? pipeline.vertexBuffers[slot] : nil
                let end = layout?.stepMode == .instance ? instanceEnd : vertexEnd
                let required = layout.map { end == 0 || $0.attributes.isEmpty ? 0 : (end - 1) * UInt64($0.stride) + $0.attributeEnd } ?? 0
                guard binding.buffer.device == device, let buffer = descriptors[binding.buffer],
                      buffer.usage.contains(.vertex), binding.offset % 4 == 0,
                      binding.offset <= buffer.size, required <= buffer.size - binding.offset else {
                    throw GPUError.validation("Invalid vertex buffer, usage, offset or fetch range")
                }
            }
        }
        func requireIndex(_ index: IndexBufferBinding, pipeline: RenderPipeline, end: UInt64 = 0) throws {
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
            guard size <= remaining, end <= size / stride else {
                throw GPUError.validation("Indexed draw exceeds index buffer range")
            }
        }
        func requireTextureUpload(
            _ destination: ResourceID, region: TextureCopyRegion, aspect: TextureViewAspect, allowsArrayLayerSpan: Bool
        ) throws -> (TextureDescriptor, TextureCopyBlock) {
            guard destination.device == device, let texture = textures[destination],
                  texture.usage.contains(.copyDestination), texture.sampleCount == 1,
                  region.fits(in: texture, allowsArrayLayerSpan: allowsArrayLayerSpan),
                  let block = texture.format.copyBlock(aspect: aspect),
                  !texture.format.hasDepth || texture.format == .depth16Unorm || aspect == .stencilOnly,
                  !texture.format.isDepthStencil || region.covers(texture) else {
                throw GPUError.validation("Invalid texture upload destination, region or aspect")
            }
            return (texture, block)
        }
        switch self {
        case .timestamped, .occlusionQueried:
            let (pass, timestamps, occlusion) = recordedPass
            guard pass.acceptsTimestampWrites else { throw GPUError.validation("Invalid query pass") }
            if let writes = timestamps {
                guard writes.querySet.device == device, let set = querySets[writes.querySet], set.type == .timestamp,
                      writes.beginningOfPassWriteIndex != writes.endOfPassWriteIndex,
                      writes.indices.allSatisfy({ $0 < set.count }) else {
                    throw GPUError.validation("Invalid timestamp pass or query indices")
                }
            }
            if let occlusion {
                guard pass.isRenderPass, occlusion.querySet.device == device,
                      let set = querySets[occlusion.querySet], set.type == .occlusion, occlusion.queryIndex < set.count else {
                    throw GPUError.validation("Invalid occlusion pass or query index")
                }
            }
            try pass.validate(device: device, descriptors: descriptors, textures: textures,
                              samplerDescriptors: samplerDescriptors, textureViews: textureViews,
                              maximumViewportDimension: maximumViewportDimension, maxVertexBuffers: maxVertexBuffers,
                              maxBindGroupsPlusVertexBuffers: maxBindGroupsPlusVertexBuffers, querySets: querySets,
                              maxMultiviewViewCount: maxMultiviewViewCount,
                              features: features, externalTextures: externalTextures, externalPlanes: externalPlanes,
                              accelerationStructures: accelerationStructures, maxComputeWorkgroupsPerDimension: maxComputeWorkgroupsPerDimension,
                              maxTaskWorkgroupTotalCount: maxTaskWorkgroupTotalCount, maxTaskWorkgroupsPerDimension: maxTaskWorkgroupsPerDimension,
                              maxMeshWorkgroupTotalCount: maxMeshWorkgroupTotalCount, maxMeshWorkgroupsPerDimension: maxMeshWorkgroupsPerDimension)
        case .resolveQuerySet(let set, let first, let count, let destination, let offset):
            guard set.device == device, let query = querySets[set], first <= query.count, count <= query.count - first,
                  destination.device == device, let buffer = descriptors[destination],
                  buffer.usage.contains(.queryResolve), offset % 256 == 0,
                  offset <= buffer.size, UInt64(count) * 8 <= buffer.size - offset else {
                throw GPUError.validation("Invalid query resolve range, destination or alignment")
            }
        case .buildTopLevelAccelerationStructure(_, let instances):
            guard instances.allSatisfy({ $0.transform.count == 12 && $0.transform.allSatisfy(\.isFinite) && $0.instanceID <= 0x00ff_ffff }) else {
                throw GPUError.validation("Invalid acceleration structure instance")
            }
        case .buildAccelerationStructure(_, let geometry):
            guard let descriptor = descriptors[geometry.vertexBuffer], descriptor.usage.contains(.vertex) else {
                throw GPUError.validation("Acceleration structure geometry requires a vertex buffer")
            }
            _ = try geometry.vertexRange(bufferSize: descriptor.size)
            if let index = geometry.indexBuffer {
                guard let buffer = descriptors[index.buffer], buffer.usage.contains(.index) else {
                    throw GPUError.validation("Acceleration structure geometry requires an index buffer")
                }
                _ = try geometry.indexRange(bufferSize: buffer.size)
            }
        case .write(let id, let offset, let words):
            try require(id, .copyDestination, UInt64(words.count) * 4, offset: offset, allowEmpty: true)
        case .writeBytes(let id, let offset, let bytes, let dataOffset, let byteCount):
            let source = try Self.uploadBytes(bytes, dataOffset: dataOffset, byteCount: byteCount)
            try require(id, .copyDestination, UInt64(source.count), offset: offset, allowEmpty: true)
        case .clearBuffer(let id, let offset, let size):
            try require(id, .copyDestination, size, offset: offset, allowEmpty: true)
        case .dispatch(let pipeline, let input, let output, let count):
            guard pipeline.device == device, pipeline.serial > 0, input != output else {
                throw GPUError.validation("Invalid pipeline identity or aliased storage buffers")
            }
            try require(input, .storage, UInt64(count) * 4)
            try require(output, .storage, UInt64(count) * 4)
        case .dispatchCompute, .dispatchComputeRanges, .dispatchComputeIndirect:
            let (pipeline, bindings, source, boundTextures, samplerBindings, immediates, structures) = computeDispatch!
            guard pipeline.id.device == device else { throw GPUError.validation("Foreign compute pipeline") }
            guard UInt64(immediates.count) == pipeline.immediateByteSize / 4 else {
                throw GPUError.validation("Compute immediate data size does not match the pipeline layout")
            }
            let indirectBuffer: ResourceID?
            switch source {
            case .direct(let groups):
                guard groups.x <= maxComputeWorkgroupsPerDimension, groups.y <= maxComputeWorkgroupsPerDimension,
                      groups.z <= maxComputeWorkgroupsPerDimension else {
                    throw GPUError.validation("Workgroup count exceeds the supported limit")
                }
                if let limit = pipeline.maximumLinearWorkgroupCount, !groups.isEmpty,
                   groups.z > limit / groups.x / groups.y {
                    throw GPUError.validation("Linear indexing dispatch exceeds u32")
                }
                indirectBuffer = nil
            case .indirect(let buffer, let offset):
                try require(buffer, .indirect, 12, offset: offset)
                indirectBuffer = buffer
            }
            guard bindings.count == pipeline.bufferBindings.count,
                  boundTextures.count == pipeline.textureBindings.count + pipeline.storageTextureBindings.count + pipeline.externalTextureBindings.count,
                  samplerBindings.count == pipeline.samplerBindings.count,
                  structures.count == pipeline.accelerationStructureBindings.count else {
                throw GPUError.validation("Compute bindings do not match the pipeline layout")
            }
            for slot in pipeline.accelerationStructureBindings {
                guard let id = structures[slot], id.device == device,
                      let descriptor = accelerationStructures[id], case .instances = descriptor.capacity else {
                    throw GPUError.validation("Invalid compute acceleration structure binding")
                }
            }
            var usedTextures = try pipeline.textureBindings.map {
                (try requireSampledTexture(boundTextures[$0.slot], $0), false)
            }
            for external in pipeline.externalTextureBindings {
                usedTextures += try requireExternalTexture(boundTextures[external]).map { ($0, false) }
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
                guard requirement.isReadOnly || binding.buffer != indirectBuffer else {
                    throw GPUError.validation("Indirect dispatch arguments alias a writable compute buffer")
                }
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
        case .copy(let source, let sourceOffset, let destination, let destinationOffset, let size):
            guard source != destination else { throw GPUError.validation("Aliased copy") }
            let count = try require(source, .copySource, size, offset: sourceOffset, allowEmpty: true)
            try require(destination, .copyDestination, count, offset: destinationOffset, allowEmpty: true)
        case .drawMesh(let pipeline, let target, let colorLoad, let workgroups, let taskBuffers, let meshBuffers, let taskTextures, let meshTextures, let taskSamplers, let meshSamplers, let additionalColors, _, let resolve, let depth, let indirect, let indirectOffset, let fragmentBuffers, let fragmentTextures, let fragmentSamplers, let raster):
            if let indirect {
                try require(indirect, .indirect, 12, offset: indirectOffset)
                guard workgroups.isEmpty else { throw GPUError.validation("Indirect mesh draw must not specify direct workgroups") }
            }
            else if indirectOffset != 0 { throw GPUError.validation("Indirect mesh offset requires an argument buffer") }
            if indirect == nil {
                let dimensionLimit = pipeline.hasTaskShader ? maxTaskWorkgroupsPerDimension : maxMeshWorkgroupsPerDimension
                let totalLimit = pipeline.hasTaskShader ? maxTaskWorkgroupTotalCount : maxMeshWorkgroupTotalCount
                let xy = workgroups.x.multipliedReportingOverflow(by: workgroups.y)
                let total = (xy.overflow ? UInt32.max : xy.partialValue).multipliedReportingOverflow(by: workgroups.z)
                guard workgroups.x <= dimensionLimit, workgroups.y <= dimensionLimit, workgroups.z <= dimensionLimit,
                      (total.overflow ? UInt32.max : total.partialValue) <= totalLimit else {
                    throw GPUError.validation("Invalid mesh draw workgroup count")
                }
            }
            guard pipeline.id.device == device, target.device == device,
                  let texture = textures[target], texture.usage.contains(.renderAttachment),
                  texture.dimension == .d2, texture.arrayLayerCount == 1, texture.sampleCount == pipeline.sampleCount,
                  texture.format == pipeline.colorFormat, texture.width > 0, texture.height > 0,
                  texture.width <= maximumViewportDimension, texture.height <= maximumViewportDimension else {
                throw GPUError.validation("Invalid mesh draw pipeline, target or workgroup count")
            }
            try raster.validate(width: texture.width, height: texture.height, maximumViewportDimension: maximumViewportDimension)
            if case .clear(let color) = colorLoad, !color.isValid(for: texture.format) {
                throw GPUError.validation("Invalid mesh draw clear color")
            }
            guard Set(additionalColors.keys) == Set(pipeline.colorFormats.keys.filter { $0 != 0 }) else {
                throw GPUError.validation("Mesh color attachments do not match pipeline")
            }
            var targets: Set<ResourceID> = [target]
            var colorViews: [TextureViewResource] = [.init(texture: target, levels: 0..<1)]
            func validateResolve(_ resolve: RenderResolveAttachment?, format: TextureFormat) throws {
                guard let resolve else { return }
                guard pipeline.sampleCount > 1, format.supportsColorResolve(features: features),
                      resolve.texture.device == device, targets.insert(resolve.texture).inserted,
                      let descriptor = textures[resolve.texture], descriptor.usage.contains(.renderAttachment),
                      descriptor.dimension == .d2, descriptor.arrayLayerCount == 1, descriptor.sampleCount == 1,
                      descriptor.format == format, let extent = descriptor.mipExtent(resolve.mipLevel),
                      extent.width == texture.width, extent.height == texture.height else {
                    throw GPUError.validation("Invalid mesh resolve attachment")
                }
                colorViews.append(.init(texture: resolve.texture, levels: resolve.mipLevel..<(resolve.mipLevel + 1)))
            }
            try validateResolve(resolve, format: texture.format)
            for (location, attachment) in additionalColors {
                guard attachment.texture.device == device, targets.insert(attachment.texture).inserted,
                      let descriptor = textures[attachment.texture], descriptor.usage.contains(.renderAttachment),
                      descriptor.dimension == .d2, descriptor.arrayLayerCount == 1, descriptor.sampleCount == pipeline.sampleCount,
                      descriptor.format == pipeline.colorFormats[location], descriptor.width == texture.width,
                      descriptor.height == texture.height, attachment.mipLevel == 0,
                      attachment.depthSlice == nil else {
                    throw GPUError.validation("Invalid mesh color attachment")
                }
                colorViews.append(.init(texture: attachment.texture, levels: 0..<1))
                try validateResolve(attachment.resolve, format: descriptor.format)
                if case .clear(let color) = attachment.loadAction, !color.isValid(for: descriptor.format) {
                    throw GPUError.validation("Invalid mesh draw clear color")
                }
            }
            guard (pipeline.depthState == nil) == (depth == nil) else { throw GPUError.validation("Mesh depth attachment does not match pipeline") }
            var depthViews: [(view: TextureViewResource, writable: Bool)] = []
            if let depth, let state = pipeline.depthState {
                guard depth.texture.device == device, let binding = textureBinding(depth.texture, usage: .renderAttachment),
                      binding.view.layers.count == 1, binding.view.dimension == .d2,
                      textureViews[depth.texture].map({ depth.mipLevel == 0 && $0.levels.count == 1 &&
                          (!binding.texture.format.hasDepth || !binding.texture.format.hasStencil || $0.aspect == .all) }) ?? true else {
                    throw GPUError.validation("Invalid mesh depth attachment view")
                }
                let descriptor = binding.texture
                let level = textureViews[depth.texture]?.levels.lowerBound ?? depth.mipLevel
                guard descriptor.sampleCount == pipeline.sampleCount,
                      descriptor.format == state.format, let extent = descriptor.mipExtent(level),
                      extent.width == texture.width, extent.height == texture.height,
                      !targets.contains(binding.view.texture), descriptor.format.hasStencil == (depth.stencil != nil),
                      descriptor.format.hasDepth == (depth.loadAction != .unused) else {
                    throw GPUError.validation("Invalid mesh depth attachment")
                }
                if case .clear(let value) = depth.loadAction, !(0...1).contains(value) { throw GPUError.validation("Invalid mesh depth clear") }
                let readOnly = depth.loadAction == .readOnly
                let stencilReadOnly = depth.stencil?.loadAction == .readOnly
                guard !(readOnly && state.writeEnabled),
                      !(stencilReadOnly && state.stencil?.writes(cullMode: pipeline.primitive.cullMode) == true),
                      !(depth.storeAction == .discard && (readOnly || depth.loadAction == .unused)),
                      !(depth.stencil?.storeAction == .discard && stencilReadOnly) else {
                    throw GPUError.validation("Invalid mesh read-only depth attachment")
                }
                if descriptor.format.hasDepth { depthViews.append((.init(texture: binding.view.texture, levels: level..<(level + 1), aspect: .depthOnly, layers: binding.view.layers), !readOnly)) }
                if descriptor.format.hasStencil { depthViews.append((.init(texture: binding.view.texture, levels: level..<(level + 1), aspect: .stencilOnly, layers: binding.view.layers), !stencilReadOnly)) }
            }
            var usedMeshTextures: [(view: TextureViewResource, writable: Bool)] = colorViews.map { ($0, true) }
            usedMeshTextures += depthViews
            for (requirements, storageRequirements, bindings) in [
                (pipeline.taskTextures, pipeline.taskStorageTextures, taskTextures),
                (pipeline.meshTextures, pipeline.meshStorageTextures, meshTextures),
                (pipeline.fragmentTextures, pipeline.fragmentStorageTextures, fragmentTextures)
            ] {
                guard requirements.count + storageRequirements.count == bindings.count else {
                    throw GPUError.validation("Mesh texture bindings do not match pipeline layout")
                }
                for requirement in requirements {
                    let view = try requireSampledTexture(bindings[requirement.slot], requirement)
                    guard !usedMeshTextures.contains(where: { $0.writable && $0.view.overlaps(view) }) else {
                        throw GPUError.validation("Aliased writable mesh textures")
                    }
                    usedMeshTextures.append((view, false))
                }
                for requirement in storageRequirements {
                    let usage: TextureUsage = requirement.access == .atomic ? .storageAtomic : .storageBinding
                    guard let binding = textureBinding(bindings[requirement.slot], usage: usage),
                          binding.texture.format == requirement.format, binding.view.levels.count == 1,
                          binding.view.dimension == requirement.dimension else {
                        throw GPUError.validation("Invalid mesh storage texture binding")
                    }
                    let view = binding.view
                    guard !usedMeshTextures.contains(where: { ($0.writable || requirement.writable) && $0.view.overlaps(view) }) else {
                        throw GPUError.validation("Aliased writable mesh textures")
                    }
                    usedMeshTextures.append((view, requirement.writable))
                }
            }
            for (requirements, bindings) in [(pipeline.taskSamplers, taskSamplers), (pipeline.meshSamplers, meshSamplers), (pipeline.fragmentSamplers, fragmentSamplers)] {
                guard requirements.count == bindings.count else { throw GPUError.validation("Mesh sampler bindings do not match pipeline layout") }
                for requirement in requirements { try requireSampler(bindings[requirement.slot], requirement) }
            }
            var used: [ResourceID: [(range: Range<UInt64>, readOnly: Bool)]] = [:]
            for (requirements, bindings) in [(pipeline.taskBufferBindings, taskBuffers), (pipeline.meshBufferBindings, meshBuffers), (pipeline.fragmentBufferBindings, fragmentBuffers)] {
                guard bindings.count == requirements.count else { throw GPUError.validation("Mesh buffer bindings do not match pipeline layout") }
                for requirement in requirements {
                    guard let binding = bindings[requirement.slot], binding.buffer.device == device,
                          let descriptor = descriptors[binding.buffer] else {
                        throw GPUError.validation("Invalid mesh buffer binding")
                    }
                    let range = try binding.range(in: descriptor, usage: requirement.usage)
                    guard requirement.isReadOnly || binding.buffer != indirect else {
                        throw GPUError.validation("Indirect mesh arguments alias a writable mesh buffer")
                    }
                    guard range.upperBound - range.lowerBound >= requirement.minimumByteSize else { throw GPUError.validation("Mesh buffer binding is too small") }
                    for previous in used[binding.buffer, default: []] {
                        if (!previous.readOnly || !requirement.isReadOnly) && previous.range.overlaps(range) {
                            throw GPUError.validation("Aliased writable mesh buffers")
                        }
                        guard previous.readOnly == requirement.isReadOnly else {
                            throw GPUError.validation("Incompatible mesh buffer access modes")
                        }
                    }
                    used[binding.buffer, default: []].append((range, requirement.isReadOnly))
                }
            }
        case .render(let pipeline, let target, let colorLoad, let count, let bindings, let firstVertex, let instances, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates, let colorStore):
            try requireRender(pipeline, target, targetMipLevel, colorLoad, targetDepthSlice: targetDepthSlice, colorStore: colorStore,
                              bindings, uniforms: uniforms, storageBuffers: storageBuffers, boundTextures: boundTextures,
                              samplerBindings: samplerBindings, vertexEnd: UInt64(firstVertex) + UInt64(count), instanceEnd: instances.end,
                              depth: depth, resolve: resolve, blendConstant: blendConstant, additionalColors: additionalColors,
                              raster: raster, immediates: immediates)
        case .renderIndexed(let pipeline, let target, let colorLoad, let index, let draw, let bindings, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates, let colorStore):
            try requireRender(pipeline, target, targetMipLevel, colorLoad, targetDepthSlice: targetDepthSlice, colorStore: colorStore,
                              bindings, uniforms: uniforms, storageBuffers: storageBuffers, indexBuffer: index.buffer,
                              boundTextures: boundTextures,
                              samplerBindings: samplerBindings, vertexEnd: 0, instanceEnd: draw.instances.end,
                              depth: depth, resolve: resolve, blendConstant: blendConstant, additionalColors: additionalColors,
                              raster: raster, immediates: immediates)
            try requireIndex(index, pipeline: pipeline, end: UInt64(draw.firstIndex) + UInt64(draw.indexCount))
        case .renderIndirect(let pipeline, let target, let colorLoad, let arguments, let offset, let index, let bindings, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates, let colorStore):
            try require(arguments, .indirect, index == nil ? 16 : 20, offset: offset)
            try requireRender(pipeline, target, targetMipLevel, colorLoad, targetDepthSlice: targetDepthSlice, colorStore: colorStore,
                              bindings, uniforms: uniforms, storageBuffers: storageBuffers, indexBuffer: index?.buffer,
                              indirectBuffer: arguments, boundTextures: boundTextures, samplerBindings: samplerBindings,
                              vertexEnd: 0, instanceEnd: 0, depth: depth, resolve: resolve, blendConstant: blendConstant,
                              additionalColors: additionalColors, raster: raster, immediates: immediates)
            if let index { try requireIndex(index, pipeline: pipeline) }
        case .copyTextureToBuffer(let source, let destination, let bytesPerRow, let mipLevel, let aspect, let offset,
                                  let layer, let z, let depth, let rowsPerImage):
            guard source.device == device, let texture = textures[source], texture.usage.contains(.copySource),
                  texture.sampleCount == 1, let extent = texture.mipPhysicalExtent(mipLevel),
                  TextureCopyRegion(width: extent.width, height: extent.height, mipLevel: mipLevel,
                                    arrayLayer: layer, z: z, depth: depth).fits(in: texture, allowsArrayLayerSpan: true),
                  let block = texture.format.copyBlock(aspect: aspect) else {
                throw GPUError.validation("Invalid texture copy source or aspect")
            }
            let size = try TextureBufferLayout(bytesPerRow: bytesPerRow, rowsPerImage: rowsPerImage)
                .byteCount(width: extent.width, height: extent.height, block: block, depth: depth)
            guard destination.device == device, let buffer = descriptors[destination], buffer.usage.contains(.copyDestination),
                  offset % UInt64(texture.format.isDepthStencil ? 4 : block.bytes) == 0, offset <= buffer.size, size <= buffer.size - offset else {
                throw GPUError.validation("Invalid texture copy buffer, offset or size")
            }
        case .copyTextureRegionToBuffer(let source, let destination, let layout, let region, let aspect):
            guard source.device == device, let texture = textures[source], texture.usage.contains(.copySource),
                  texture.sampleCount == 1, region.fits(in: texture, allowsArrayLayerSpan: true),
                  let block = texture.format.copyBlock(aspect: aspect),
                  !texture.format.isDepthStencil || region.covers(texture) else {
                throw GPUError.validation("Invalid texture copy source or aspect")
            }
            let size = try layout.byteCount(width: region.width, height: region.height, block: block, depth: region.depth)
            guard destination.device == device, let buffer = descriptors[destination], buffer.usage.contains(.copyDestination),
                  layout.offset % UInt64(texture.format.isDepthStencil ? 4 : block.bytes) == 0,
                  layout.offset <= buffer.size, size <= buffer.size - layout.offset else {
                throw GPUError.validation("Invalid texture copy buffer, offset or size")
            }
        case .copyBufferToTexture(let source, let layout, let destination, let region, let aspect):
            let (texture, block) = try requireTextureUpload(destination, region: region, aspect: aspect, allowsArrayLayerSpan: true)
            let count = try layout.byteCount(width: region.width, height: region.height, block: block, depth: region.depth)
            guard source.device == device, let buffer = descriptors[source], buffer.usage.contains(.copySource),
                  layout.offset % UInt64(texture.format.isDepthStencil ? 4 : block.bytes) == 0,
                  layout.offset <= buffer.size, count <= buffer.size - layout.offset else {
                throw GPUError.validation("Invalid texture upload buffer, offset or size")
            }
        case .writeTexture(let destination, let data, let layout, let region, let aspect):
            let (_, block) = try requireTextureUpload(destination, region: region, aspect: aspect, allowsArrayLayerSpan: true)
            _ = try layout.footprint(region: region, block: block, dataSize: UInt64(data.count))
        case .copyTextureToTexture(let source, let destination, let region, let origin):
            let destinationRegion = origin.region(width: region.width, height: region.height, depth: region.depth)
            guard source.device == device, destination.device == device,
                  let src = textures[source], let dst = textures[destination],
                  src.usage.contains(.copySource), dst.usage.contains(.copyDestination),
                  src.format.copyCompatibleFormat == dst.format.copyCompatibleFormat, src.sampleCount == dst.sampleCount,
                  region.fits(in: src, allowsArrayLayerSpan: true),
                  destinationRegion.fits(in: dst, allowsArrayLayerSpan: true) else {
                throw GPUError.validation("Invalid texture copy resources, formats, sample counts or regions")
            }
            if source == destination && region.mipLevel == origin.mipLevel {
                guard !(src.dimension == .d3 && region.depth == 0 && region.z == destinationRegion.z),
                      !region.planes(in: src).overlaps(destinationRegion.planes(in: dst)) else {
                    throw GPUError.validation("Texture copy subresources must be disjoint")
                }
            }
            if src.format.isDepthStencil || src.sampleCount > 1 {
                guard region.covers(src), destinationRegion.covers(dst) else {
                    throw GPUError.validation("Depth and multisampled texture copies require full mip extents")
                }
            }
        case .read(let id, let size):
            try require(id, .mapRead, size)
        }
    }
}
