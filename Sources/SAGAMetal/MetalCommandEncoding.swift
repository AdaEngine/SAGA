import SAGABackend
import SAGAShaderIR
import SAGATypes
#if canImport(Metal)
import Metal
import Darwin

extension MetalDevice {
    func encodeCommands(
        _ commands: ArraySlice<BufferCommand>, resources: [ResourceID],
        initialized: inout [ResourceID: Set<TextureSubresource>], writtenQueries: inout [ResourceID: Set<UInt32>],
        precedingAccelerationBuilds: [ResourceID: PendingAccelerationBuild]
    ) throws -> (Batch, [(any MTLBuffer, Int)]) {
        let trace = Self.traceGrouping
        let cpuStart = trace ? clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) : 0
        var stats = GroupingStats()
        guard let commandBuffer = queue.makeCommandBuffer() else { throw GPUError.outOfMemory }
        var retained = resources.compactMap { buffers[$0]?.native ?? querySets[$0]?.visibility }
        var retainedCopyViews: [any MTLTexture] = []
        var retainedPipelines: [any MTLComputePipelineState] = []
        var retainedRenderPipelines: [RenderState] = []
        var retainedMeshPipelines: [any MTLRenderPipelineState] = []
        var committedHandlers: [@MainActor @Sendable () -> Void] = []
        var reads: [(any MTLBuffer, Int)] = []
        var accelerationBuilds: [ResourceID: PendingAccelerationBuild] = [:]
        var retainedAccelerationBuilds: [MetalAccelerationStructureBuild] = []
        func encodeDraw(
            _ pipeline: RenderPipeline, target: ResourceID?, colorLoad: RenderColorLoadAction, colorStore: RenderStoreAction,
            level: UInt32, layer: UInt32?,
            depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?, additionalColors: [UInt32: RenderColorAttachment],
            vertexBuffers: [VertexBufferBinding], uniforms: [ComputeBindingSlot: ComputeBufferBinding],
            storageBuffers: [ComputeBindingSlot: ComputeBufferBinding], textures: [ComputeBindingSlot: ResourceID],
            samplers: [ComputeBindingSlot: ResourceID], blendConstant: RenderClearColor, raster: RenderRasterState,
            immediates: [UInt32], drawEnabled: Bool, timestamps: PassTimestampWrites?, occlusion: PassOcclusionQuery?,
            draw: (any MTLRenderCommandEncoder) -> Void
        ) throws {
            let state = renderPipelines[pipeline.id]!
            let rateBinding = raster.rateMap as? MetalRasterizationRateBinding
            guard (raster.rateMap == nil || rateBinding != nil), state.metalRenderTargetLayer == rateBinding?.vertexLayer else {
                throw GPUError.validation("Metal rate map layer does not match the vertex stage")
            }
            if let rateBinding {
                let attachment = target.map { ($0, level) }
                    ?? additionalColors.values.first.map { ($0.texture, $0.mipLevel) }
                    ?? depth.map { ($0.texture, $0.mipLevel) }
                guard rateBinding.map.device.registryID == native.registryID,
                      native.supportsRasterizationRateMap(layerCount: Int(rateBinding.layerCount)),
                      let (id, mip) = attachment,
                      max(1, nativeTexture(id).width >> Int(mip)) >= rateBinding.physicalSize.width,
                      max(1, nativeTexture(id).height >> Int(mip)) >= rateBinding.physicalSize.height else {
                    throw GPUError.validation("Metal rate map cannot rasterize to the supplied attachments")
                }
            }
            if drawEnabled { try initializeBoundTextures(textures, on: commandBuffer, initialized: &initialized) }
            if rateBinding != nil {
                // Initialize the entire allocation, including padding outside the physical rate-map region.
                let attachments = [target, depth?.texture].compactMap { $0 } + additionalColors.values.map(\.texture)
                for id in attachments {
                    try initializeTextureSubresources(id, on: commandBuffer, initialized: &initialized)
                }
            } else {
                try initializeLoadedAttachments(target: target, level: level, depthSlice: layer, colorLoad: colorLoad, depth: depth,
                                                additionalColors: additionalColors, on: commandBuffer, initialized: &initialized)
            }
            let vertexLengths = drawEnabled ? try runtimeLengthValues(state.vertexRuntimeLengths, bindings: storageBuffers) : []
            let fragmentLengths = drawEnabled ? try runtimeLengthValues(state.fragmentRuntimeLengths, bindings: storageBuffers) : []
            let clear = colorLoad.clearColor.map { MTLClearColor(red: $0.red, green: $0.green, blue: $0.blue, alpha: $0.alpha) }
            var viewCount = 1
            var viewLayer: UInt32 = 0
            var viewMappings: [MTLVertexAmplificationViewMapping] = []
            if let mask = pipeline.multiviewMask {
                let attachment = target ?? additionalColors.values.first?.texture ?? depth!.texture
                viewCount = renderAttachmentLayers(attachment).count
                if mask.nonzeroBitCount == 1 {
                    viewLayer = UInt32(mask.trailingZeroBitCount)
                    if viewCount > 1 {
                        try clearSingleViewAttachments(commandBuffer, target: target, level: level, clear: clear,
                                                       depth: depth, additionalColors: additionalColors, viewCount: viewCount)
                    }
                } else {
                    for layer in 0..<viewCount where mask & (1 << layer) != 0 {
                        viewMappings.append(.init(viewportArrayIndexOffset: 0, renderTargetArrayIndexOffset: UInt32(layer)))
                    }
                }
            }
            var visibility: (any MTLBuffer)?
            if let occlusion {
                let buffer = querySets[occlusion.querySet]!.visibility!
                let offset = Int(occlusion.queryIndex) * 8
                guard let reset = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                reset.fill(buffer: buffer, range: offset..<(offset + 8), value: 0)
                reset.endEncoding()
                if drawEnabled { visibility = buffer }
            }
            if !drawEnabled, let timestamps, let start = timestamps.beginningOfPassWriteIndex {
                retained.append(try encodeTimestampMarker(commandBuffer, writes: .init(
                    querySet: timestamps.querySet, beginningOfPassWriteIndex: start)))
            }
            let deferResolve = viewCount > 1 && viewMappings.isEmpty &&
                (resolve != nil || additionalColors.values.contains(where: { $0.resolve != nil }))
            let encoder = try renderEncoder(commandBuffer, target: target.map { nativeTexture($0) }, level: level, layer: layer ?? 0,
                                            clear: clear, depth: depth, resolve: resolve, additionalColors: additionalColors,
                                            colorStore: colorStore, timestamps: drawEnabled ? timestamps : nil, visibility: visibility,
                                            viewCount: viewMappings.isEmpty ? 1 : viewCount, viewLayer: viewLayer,
                                            deferResolve: deferResolve, rateMap: rateBinding?.map)
            if drawEnabled {
                if !viewMappings.isEmpty {
                    viewMappings.withUnsafeBufferPointer {
                        encoder.setVertexAmplificationCount($0.count, viewMappings: $0.baseAddress)
                    }
                }
                bindRender(encoder, state: state, pipeline: pipeline, vertexBuffers: vertexBuffers, uniforms: uniforms,
                           storageBuffers: storageBuffers, vertexLengths: vertexLengths, fragmentLengths: fragmentLengths,
                           boundTextures: textures, samplerBindings: samplers, blendConstant: blendConstant, raster: raster)
                if let rateBinding, raster.viewport == nil {
                    let size = rateBinding.map.screenSize
                    encoder.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(size.width),
                                                    height: Double(size.height), znear: 0, zfar: 1))
                }
                if !immediates.isEmpty {
                    immediates.withUnsafeBytes { bytes in
                        if let index = state.vertexImmediateBufferIndex {
                            encoder.setVertexBytes(bytes.baseAddress!, length: bytes.count, index: Int(index))
                        }
                        if let index = state.fragmentImmediateBufferIndex {
                            encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: Int(index))
                        }
                    }
                }
                if let stencil = depth?.stencil { encoder.setStencilReferenceValue(stencil.reference & 0xff) }
                if let occlusion { encoder.setVisibilityResultMode(.boolean, offset: Int(occlusion.queryIndex) * 8) }
                draw(encoder)
                if occlusion != nil { encoder.setVisibilityResultMode(.disabled, offset: 0) }
            }
            encoder.endEncoding()
            if deferResolve {
                try resolveLayeredAttachments(commandBuffer, target: target, level: level, resolve: resolve,
                                              colorStore: colorStore, additionalColors: additionalColors, viewCount: viewCount)
            }
            if !drawEnabled, let timestamps, let end = timestamps.endOfPassWriteIndex {
                retained.append(try encodeTimestampMarker(commandBuffer, writes: .init(
                    querySet: timestamps.querySet, endOfPassWriteIndex: end)))
            }
            try clearDiscardedAttachments(commandBuffer, target: target, level: level, layer: layer ?? 0,
                                          colorStore: colorStore, depth: depth, additionalColors: additionalColors,
                                          viewCount: viewCount)
            retainedRenderPipelines.append(state)
            markRenderedAttachments(target: target, level: level, depthSlice: layer, depth: depth, resolve: resolve,
                                    additionalColors: additionalColors, initialized: &initialized)
        }

        var groupedUntil = commands.startIndex
        for commandIndex in commands.indices {
            if commandIndex < groupedUntil { continue }
            if commandIndex + 1 < commands.endIndex,
               let first = indexedColorDraw(commands[commandIndex], index: commandIndex),
               let second = indexedColorDraw(commands[commandIndex + 1], index: commandIndex + 1), second.follows(first) {
                var group = [first, second]
                var next = commandIndex + 2
                while next < commands.endIndex, let draw = indexedColorDraw(commands[next], index: next), draw.follows(first) {
                    group.append(draw)
                    next += 1
                }
                try encodeIndexedColorGroup(group[...], first: first, on: commandBuffer, initialized: &initialized,
                                                          retained: &retained, pipelines: &retainedRenderPipelines)
                if trace {
                    stats.indexedDraws += group.count
                    stats.groupedDraws += group.count
                    stats.groups += 1
                    stats.renderEncoders += 1
                    stats.indexEncoders += 1
                }
                groupedUntil = next
                continue
            }
            let recorded = commands[commandIndex]
            let (command, timestamps, occlusion) = recorded.recordedPass
            if let timestamps { writtenQueries[timestamps.querySet, default: []].formUnion(timestamps.indices) }
            if let occlusion { writtenQueries[occlusion.querySet, default: []].insert(occlusion.queryIndex) }
            switch command {
            case let .native(pass):
                try committedHandlers.append(contentsOf: recordNativePass(pass, on: commandBuffer, initialized: &initialized))
            case .drawMesh(let pipeline, let target, let colorLoad, let workgroups, let taskBuffers, let meshBuffers, let taskTextures, let meshTextures, let taskSamplers, let meshSamplers, let additionalColors, let colorStore, let resolve, let depth, let indirectBuffer, let indirectOffset, let fragmentBuffers, let fragmentTextures, let fragmentSamplers, let raster):
                guard #available(macOS 13, iOS 16, *) else { throw GPUError.unsupported("Metal mesh shaders are unavailable") }
                let state = meshPipelines[pipeline.id]!
                let indirect: (any MTLBuffer)?
                if let indirectBuffer {
                    if meshDispatchValidator == nil {
                        meshDispatchValidator = try MetalDispatchValidator(device: native)
                    }
                    let axisLimit = pipeline.hasTaskShader ? capabilities.maxTaskWorkgroupsPerDimension : capabilities.maxMeshWorkgroupsPerDimension
                    let totalLimit = pipeline.hasTaskShader ? capabilities.maxTaskWorkgroupTotalCount : capabilities.maxMeshWorkgroupTotalCount
                    let validated = try meshDispatchValidator!.encode(on: commandBuffer, device: native, source: buffers[indirectBuffer]!.native,
                        offset: Int(indirectOffset), maxWorkgroupsPerDimension: axisLimit, maximumLinearWorkgroupCount: totalLimit)
                    retained.append(validated)
                    indirect = validated
                } else { indirect = nil }
                let drawEnabled = indirect != nil || (workgroups.x > 0 && workgroups.y > 0 && workgroups.z > 0)
                if drawEnabled {
                    try initializeBoundTextures(taskTextures, on: commandBuffer, initialized: &initialized)
                    try initializeBoundTextures(meshTextures, on: commandBuffer, initialized: &initialized)
                    try initializeBoundTextures(fragmentTextures, on: commandBuffer, initialized: &initialized)
                }
                try initializeLoadedAttachments(target: target, level: 0, depthSlice: nil, colorLoad: colorLoad, depth: depth,
                                                additionalColors: additionalColors, on: commandBuffer, initialized: &initialized)
                let clear = colorLoad.clearColor.map { MTLClearColor(red: $0.red, green: $0.green, blue: $0.blue, alpha: $0.alpha) }
                var visibility: (any MTLBuffer)?
                if let occlusion {
                    let buffer = querySets[occlusion.querySet]!.visibility!
                    let offset = Int(occlusion.queryIndex) * 8
                    guard let reset = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                    reset.fill(buffer: buffer, range: offset..<(offset + 8), value: 0)
                    reset.endEncoding()
                    if drawEnabled { visibility = buffer }
                }
                if !drawEnabled, let timestamps, let start = timestamps.beginningOfPassWriteIndex {
                    retained.append(try encodeTimestampMarker(commandBuffer, writes: .init(querySet: timestamps.querySet, beginningOfPassWriteIndex: start)))
                }
                let encoder = try renderEncoder(commandBuffer, target: nativeTexture(target), clear: clear, depth: depth, resolve: resolve, additionalColors: additionalColors, colorStore: colorStore, timestamps: drawEnabled ? timestamps : nil, visibility: visibility)
                if drawEnabled {
                    if let occlusion { encoder.setVisibilityResultMode(.boolean, offset: Int(occlusion.queryIndex) * 8) }
                    encoder.setRenderPipelineState(state.pipeline)
                    if let viewport = raster.viewport {
                        encoder.setViewport(MTLViewport(originX: Double(viewport.x), originY: Double(viewport.y), width: Double(viewport.width), height: Double(viewport.height), znear: Double(viewport.minDepth), zfar: Double(viewport.maxDepth)))
                    }
                    if let scissor = raster.scissor {
                        encoder.setScissorRect(MTLScissorRect(x: Int(scissor.x), y: Int(scissor.y), width: Int(scissor.width), height: Int(scissor.height)))
                    }
                    encoder.setFrontFacing(pipeline.primitive.frontFace == .clockwise ? .clockwise : .counterClockwise)
                    if capabilities.features.contains(.depthClipControl) { encoder.setDepthClipMode(pipeline.primitive.unclippedDepth ? .clamp : .clip) }
                    switch pipeline.primitive.cullMode {
                    case .none: encoder.setCullMode(.none)
                    case .front: encoder.setCullMode(.front)
                    case .back: encoder.setCullMode(.back)
                    }
                    if state.writesDepth {
                        var depthRange = SIMD2<Float>(raster.viewport?.minDepth ?? 0, raster.viewport?.maxDepth ?? 1)
                        encoder.setFragmentBytes(&depthRange, length: MemoryLayout.size(ofValue: depthRange),
                                                 index: RenderStageArtifact.metalViewportDepthBufferIndex)
                    }
                    if let depthState = state.depth { encoder.setDepthStencilState(depthState) }
                    if let stencil = depth?.stencil { encoder.setStencilReferenceValue(stencil.reference & 0xff) }
                    if let bias = pipeline.depthState?.bias, bias != .init() { encoder.setDepthBias(Float(bias.constant), slopeScale: bias.slopeScale, clamp: bias.clamp) }
                    for (slot, index) in state.objectUniforms + state.objectStorage {
                        let binding = taskBuffers[slot]!
                        encoder.setObjectBuffer(buffers[binding.buffer]!.native, offset: Int(binding.offset), index: index)
                    }
                    for (slot, index) in state.meshUniforms + state.meshStorage {
                        let binding = meshBuffers[slot]!
                        encoder.setMeshBuffer(buffers[binding.buffer]!.native, offset: Int(binding.offset), index: index)
                    }
                    for (slot, index) in state.objectTextures { encoder.setObjectTexture(nativeTexture(taskTextures[slot]!), index: index) }
                    for (slot, index) in state.meshTextures { encoder.setMeshTexture(nativeTexture(meshTextures[slot]!), index: index) }
                    for (slot, index) in state.objectSamplers { encoder.setObjectSamplerState(samplers[taskSamplers[slot]!]!.native, index: index) }
                    for (slot, index) in state.meshSamplers { encoder.setMeshSamplerState(samplers[meshSamplers[slot]!]!.native, index: index) }
                    for (slot, index) in state.fragmentBuffers {
                        let binding = fragmentBuffers[slot]!
                        encoder.setFragmentBuffer(buffers[binding.buffer]!.native, offset: Int(binding.offset), index: index)
                    }
                    for (slot, index) in state.fragmentTextures { encoder.setFragmentTexture(nativeTexture(fragmentTextures[slot]!), index: index) }
                    for binding in pipeline.taskStorageTextures where binding.access == .atomic {
                        encoder.useResource(nativeTexture(taskTextures[binding.slot]!), usage: [.read, .write], stages: .object)
                    }
                    for binding in pipeline.meshStorageTextures where binding.access == .atomic {
                        encoder.useResource(nativeTexture(meshTextures[binding.slot]!), usage: [.read, .write], stages: .mesh)
                    }
                    for binding in pipeline.fragmentStorageTextures where binding.access == .atomic {
                        encoder.useResource(nativeTexture(fragmentTextures[binding.slot]!), usage: [.read, .write], stages: .fragment)
                    }
                    if let index = state.fragmentTextureAtomicGuardBufferIndex {
                        var disabled: UInt32 = 0
                        encoder.setFragmentBytes(&disabled, length: MemoryLayout<UInt32>.size, index: Int(index))
                    }
                    for (slot, index) in state.fragmentSamplers { encoder.setFragmentSamplerState(samplers[fragmentSamplers[slot]!]!.native, index: index) }
                    let fragmentLengths = try runtimeLengthValues(state.fragmentLengths, bindings: fragmentBuffers)
                    if !fragmentLengths.isEmpty {
                        fragmentLengths.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: Int(RenderStageArtifact.metalStorageLengthBufferIndex)) }
                    }
                    let objectLengths = try runtimeLengthValues(state.objectLengths, bindings: taskBuffers)
                    let meshLengths = try runtimeLengthValues(state.meshLengths, bindings: meshBuffers)
                    let lengthIndex = Int(ComputeShaderArtifact.storageBindingCapacity)
                    if !objectLengths.isEmpty {
                        objectLengths.withUnsafeBytes { encoder.setObjectBytes($0.baseAddress!, length: $0.count, index: lengthIndex) }
                    }
                    if !meshLengths.isEmpty {
                        meshLengths.withUnsafeBytes { encoder.setMeshBytes($0.baseAddress!, length: $0.count, index: lengthIndex) }
                    }
                    if let indirect {
                        encoder.drawMeshThreadgroups(indirectBuffer: indirect, indirectBufferOffset: 0,
                            threadsPerObjectThreadgroup: state.objectThreads, threadsPerMeshThreadgroup: state.meshThreads)
                    } else {
                        encoder.drawMeshThreadgroups(MTLSize(width: Int(workgroups.x), height: Int(workgroups.y), depth: Int(workgroups.z)),
                            threadsPerObjectThreadgroup: state.objectThreads, threadsPerMeshThreadgroup: state.meshThreads)
                    }
                }
                if drawEnabled && occlusion != nil { encoder.setVisibilityResultMode(.disabled, offset: 0) }
                encoder.endEncoding()
                if !drawEnabled, let timestamps, let end = timestamps.endOfPassWriteIndex {
                    retained.append(try encodeTimestampMarker(commandBuffer, writes: .init(querySet: timestamps.querySet, endOfPassWriteIndex: end)))
                }
                retainedMeshPipelines.append(state.pipeline)
                markRenderedAttachments(target: target, level: 0, depthSlice: nil, depth: depth, resolve: resolve,
                                        additionalColors: additionalColors, initialized: &initialized)
                try clearDiscardedAttachments(commandBuffer, target: target, level: 0, layer: 0, colorStore: colorStore, depth: depth, additionalColors: additionalColors, viewCount: 1)
                if let timestamps { writtenQueries[timestamps.querySet, default: []].formUnion(timestamps.indices) }
                if let occlusion { writtenQueries[occlusion.querySet, default: []].insert(occlusion.queryIndex) }
            case .buildAccelerationStructure(let destination, let geometry):
                let build = try MetalAccelerationStructureBuild.encode(device: native, commandBuffer: commandBuffer,
                    vertexBuffer: buffers[geometry.vertexBuffer]!.native, vertexOffset: Int(geometry.vertexOffset),
                    vertexStride: Int(geometry.vertexStride), vertexCount: Int(geometry.vertexCount),
                    indices: geometry.indexBuffer.map { (buffers[$0.buffer]!.native, $0, geometry.indexCount) })
                retainedAccelerationBuilds.append(build)
                guard nextAccelerationBuild < UInt64.max else { throw GPUError.outOfMemory }
                nextAccelerationBuild += 1
                accelerationBuilds[destination] = PendingAccelerationBuild(serial: nextAccelerationBuild, build: build.persistent)
            case .renderMultiIndirect:
                throw GPUError.unsupported("Metal does not run multi-draw indirect commands")
            case .buildBoxAccelerationStructure:
                throw GPUError.unsupported("Metal does not build box acceleration structures")
            case .buildTopLevelAccelerationStructure(let destination, let instances):
                let inputs = try instances.map { instance in
                    guard let source = accelerationBuilds[instance.accelerationStructure]
                        ?? precedingAccelerationBuilds[instance.accelerationStructure]
                        ?? accelerationStructures[instance.accelerationStructure]?.build else {
                        throw GPUError.validation("Top-level acceleration structure references an unbuilt BLAS")
                    }
                    return (instance, source.build)
                }
                let build = try MetalAccelerationStructureBuild.encodeTopLevel(device: native,
                    commandBuffer: commandBuffer, instances: inputs)
                retainedAccelerationBuilds.append(build)
                guard nextAccelerationBuild < UInt64.max else { throw GPUError.outOfMemory }
                nextAccelerationBuild += 1
                accelerationBuilds[destination] = PendingAccelerationBuild(serial: nextAccelerationBuild, build: build.persistent)
            case .timestamped, .occlusionQueried:
                throw GPUError.validation("Invalid query pass nesting")
            case .resolveQuerySet(let id, let first, let count, let destination, let offset):
                if count == 0 { continue }
                let query = querySets[id]!
                try resolveQueries(query, written: query.written.union(writtenQueries[id, default: []]),
                                   range: first..<(first + count), destination: buffers[destination]!.native,
                                   offset: Int(offset), on: commandBuffer)
            case .write(let id, let offset, let words):
                if words.isEmpty { continue }
                retained.append(try words.withUnsafeBytes {
                    try encodeUpload($0, to: buffers[id]!.native, offset: Int(offset), on: commandBuffer)
                })
            case .writeBytes(let id, let offset, let bytes, let dataOffset, let byteCount):
                let source = try BufferCommand.uploadBytes(bytes, dataOffset: dataOffset, byteCount: byteCount)
                if source.isEmpty { continue }
                retained.append(try source.withUnsafeBytes {
                    try encodeUpload($0, to: buffers[id]!.native, offset: Int(offset), on: commandBuffer)
                })
            case .clearBuffer(let id, let offset, let count):
                let buffer = buffers[id]!
                let size = count ?? (buffer.descriptor.size - offset)
                if size == 0 { continue }
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.fill(buffer: buffer.native, range: Int(offset)..<Int(offset + size), value: 0)
                encoder.endEncoding()
            case .dispatch(let id, let input, let output, let count):
                let pipeline = pipelines[id]!
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(buffers[input]!.native, offset: 0, index: 0)
                encoder.setBuffer(buffers[output]!.native, offset: 0, index: 1)
                var count = count
                encoder.setBytes(&count, length: 4, index: 2)
                let width = min(64, pipeline.maxTotalThreadsPerThreadgroup)
                let groups = (Int(count) + width - 1) / width
                encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                encoder.endEncoding()
                retainedPipelines.append(pipeline)
            case .dispatchCompute, .dispatchComputeRanges, .dispatchComputeIndirect:
                let (pipeline, bindings, source, boundTextures, samplerBindings, immediates, boundAccelerationStructures) = command.computeDispatch!
                if case .direct(let workgroups) = source, workgroups.isEmpty {
                    if let timestamps {
                        retained.append(try encodeTimestampMarker(commandBuffer, writes: timestamps))
                    }
                    continue
                }
                let compute = computePipelines[pipeline.id]!
                let lengths = try runtimeLengthValues(compute.runtimeLengths, bindings: bindings)
                try initializeBoundTextures(boundTextures, on: commandBuffer, initialized: &initialized)
                let indirect: (any MTLBuffer)?
                if case .indirect(let buffer, let offset) = source {
                    if dispatchValidator == nil {
                        dispatchValidator = try MetalDispatchValidator(device: native)
                    }
                    let validated = try dispatchValidator!.encode(
                        on: commandBuffer, device: native, source: buffers[buffer]!.native, offset: Int(offset),
                        maxWorkgroupsPerDimension: capabilities.maxComputeWorkgroupsPerDimension,
                        maximumLinearWorkgroupCount: pipeline.maximumLinearWorkgroupCount)
                    retained.append(validated)
                    indirect = validated
                } else {
                    indirect = nil
                }
                let encoder = try computeEncoder(commandBuffer, timestamps: timestamps)
                defer { encoder.endEncoding() }
                if indirect != nil && timestamps != nil {
                    retained.append(try encodeTimestampWork(encoder))
                }
                encoder.setComputePipelineState(compute.state)
                for (slot, index) in compute.textureIndices {
                    encoder.setTexture(nativeTexture(boundTextures[slot]!), index: index)
                }
                // Declare atomic writes so Metal tracks them before subsequent blit reads.
                for binding in pipeline.storageTextureBindings where binding.access == .atomic {
                    encoder.useResource(nativeTexture(boundTextures[binding.slot]!), usage: [.read, .write])
                }
                bindExternalTextures(compute.externalTextures, bindings: boundTextures,
                                     setPlane: { encoder.setTexture($0, index: $1) },
                                     setParameters: { encoder.setBytes($0, length: $1, index: $2) })
                for (slot, index) in compute.samplerIndices {
                    encoder.setSamplerState(samplers[samplerBindings[slot]!]!.native, index: index)
                }
                for (slot, index) in compute.bufferIndices {
                    let binding = bindings[slot]!
                    encoder.setBuffer(buffers[binding.buffer]!.native, offset: Int(binding.offset),
                                      index: index)
                }
                for (slot, index) in compute.accelerationIndices {
                    let id = boundAccelerationStructures[slot]!
                    guard let build = accelerationBuilds[id] ?? precedingAccelerationBuilds[id]
                        ?? accelerationStructures[id]?.build else {
                        throw GPUError.validation("Metal compute dispatch references an unbuilt TLAS")
                    }
                    encoder.setAccelerationStructure(build.build.structure, bufferIndex: index)
                    for dependency in build.build.dependencies {
                        encoder.useResource(dependency.structure, usage: .read)
                    }
                    retainedAccelerationBuilds.append(build.build)
                }
                if let index = compute.lengthBufferIndex {
                    lengths.withUnsafeBytes {
                        encoder.setBytes($0.baseAddress!, length: $0.count, index: Int(index))
                    }
                }
                if let index = compute.immediateBufferIndex {
                    immediates.withUnsafeBytes {
                        encoder.setBytes($0.baseAddress!, length: $0.count, index: Int(index))
                    }
                }
                switch source {
                case .direct(let workgroups):
                    encoder.dispatchThreadgroups(
                        MTLSize(width: Int(workgroups.x), height: Int(workgroups.y), depth: Int(workgroups.z)),
                        threadsPerThreadgroup: compute.threadsPerThreadgroup)
                case .indirect:
                    encoder.dispatchThreadgroups(indirectBuffer: indirect!, indirectBufferOffset: 0,
                                                 threadsPerThreadgroup: compute.threadsPerThreadgroup)
                }
                retainedPipelines.append(compute.state)
            case .copy(let source, let sourceOffset, let destination, let destinationOffset, let byteCount):
                let input = buffers[source]!
                let size = byteCount ?? (input.descriptor.size - sourceOffset)
                if size == 0 { continue }
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.copy(from: input.native, sourceOffset: Int(sourceOffset),
                             to: buffers[destination]!.native, destinationOffset: Int(destinationOffset), size: Int(size))
                encoder.endEncoding()
            case .render(let pipeline, let target, let colorLoad, let vertexCount, let vertexBuffers, let firstVertex, let instances, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates, let colorStore):
                let drawEnabled = vertexCount > 0 && instances.count > 0 && !raster.hasEmptyScissor
                try encodeDraw(pipeline, target: target, colorLoad: colorLoad, colorStore: colorStore, level: targetMipLevel, layer: targetDepthSlice,
                               depth: depth, resolve: resolve, additionalColors: additionalColors, vertexBuffers: vertexBuffers,
                               uniforms: uniforms, storageBuffers: storageBuffers, textures: boundTextures, samplers: samplerBindings,
                               blendConstant: blendConstant, raster: raster, immediates: immediates, drawEnabled: drawEnabled,
                               timestamps: timestamps, occlusion: occlusion) { encoder in
                    encoder.drawPrimitives(type: pipeline.primitive.topology.metal.type, vertexStart: Int(firstVertex),
                                           vertexCount: Int(vertexCount), instanceCount: Int(instances.count),
                                           baseInstance: Int(instances.first))
                }
            case .renderIndexed(let pipeline, let target, let colorLoad, let index, let draw, let vertexBuffers, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates, let colorStore):
                if trace {
                    stats.indexedDraws += 1
                    stats.renderEncoders += 1
                }
                let indexBuffer = buffers[index.buffer]!.native
                let byteOffset = Int(index.offset + UInt64(draw.firstIndex) * UInt64(index.format.byteSize))
                var indirect: (any MTLBuffer)?
                var drawEnabled = draw.indexCount > 0 && draw.instances.count > 0 && !raster.hasEmptyScissor
                if drawEnabled && pipeline.vertexBuffers.contains(where: { $0.stepMode == .vertex && !$0.attributes.isEmpty }) {
                    let capacity = fetchCapacity(pipeline.vertexBuffers, bindings: vertexBuffers, stepMode: .vertex)
                    if capacity > 0 {
                        if indexValidator == nil { indexValidator = try MetalIndexValidator(device: native) }
                        let args = try indexValidator!.encode(
                            on: commandBuffer, device: native, indices: indexBuffer, format: index.format,
                            byteOffset: byteOffset, draw: draw, maximumVertex: UInt32(capacity - 1),
                            primitiveRestart: pipeline.primitive.topology.isStrip)
                        if trace { stats.indexEncoders += 1 }
                        indirect = args
                        retained.append(args)
                    } else {
                        drawEnabled = false
                    }
                }
                try encodeDraw(pipeline, target: target, colorLoad: colorLoad, colorStore: colorStore, level: targetMipLevel, layer: targetDepthSlice,
                               depth: depth, resolve: resolve, additionalColors: additionalColors, vertexBuffers: vertexBuffers,
                               uniforms: uniforms, storageBuffers: storageBuffers, textures: boundTextures, samplers: samplerBindings,
                               blendConstant: blendConstant, raster: raster, immediates: immediates, drawEnabled: drawEnabled,
                               timestamps: timestamps, occlusion: occlusion) { encoder in
                    if let indirect {
                        encoder.drawIndexedPrimitives(type: pipeline.primitive.topology.metal.type, indexType: index.format.metal,
                                                      indexBuffer: indexBuffer, indexBufferOffset: byteOffset,
                                                      indirectBuffer: indirect, indirectBufferOffset: 0)
                    } else {
                        encoder.drawIndexedPrimitives(type: pipeline.primitive.topology.metal.type, indexCount: Int(draw.indexCount),
                                                      indexType: index.format.metal, indexBuffer: indexBuffer,
                                                      indexBufferOffset: byteOffset, instanceCount: Int(draw.instances.count),
                                                      baseVertex: Int(draw.baseVertex), baseInstance: Int(draw.instances.first))
                    }
                }
            case .renderIndirect(let pipeline, let target, let colorLoad, let arguments, let offset, let index, let vertexBuffers, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates, let colorStore):
                let indices = index.map { binding in
                    let indexBuffer = buffers[binding.buffer]!
                    let byteCount = binding.size ?? (indexBuffer.descriptor.size - binding.offset)
                    return (binding: binding, buffer: indexBuffer.native, count: byteCount / UInt64(binding.format.byteSize))
                }
                let vertices = fetchCapacity(pipeline.vertexBuffers, bindings: vertexBuffers, stepMode: .vertex)
                let instances = fetchCapacity(pipeline.vertexBuffers, bindings: vertexBuffers, stepMode: .instance)
                let drawEnabled = vertices > 0 && instances > 0 && (indices?.count ?? 1) > 0 && !raster.hasEmptyScissor
                var validated: (any MTLBuffer)?
                if drawEnabled {
                    if indirectDrawValidator == nil { indirectDrawValidator = try MetalIndirectDrawValidator(device: native) }
                    let args = try indirectDrawValidator!.encode(
                        on: commandBuffer, device: native, source: buffers[arguments]!.native, offset: Int(offset),
                        vertices: vertices, instances: instances,
                        indices: indices, scanIndices: pipeline.vertexBuffers.contains { $0.stepMode == .vertex && !$0.attributes.isEmpty },
                        primitiveRestart: pipeline.primitive.topology.isStrip)
                    retained.append(args)
                    validated = args
                }
                try encodeDraw(pipeline, target: target, colorLoad: colorLoad, colorStore: colorStore, level: targetMipLevel, layer: targetDepthSlice,
                               depth: depth, resolve: resolve, additionalColors: additionalColors, vertexBuffers: vertexBuffers,
                               uniforms: uniforms, storageBuffers: storageBuffers, textures: boundTextures, samplers: samplerBindings,
                               blendConstant: blendConstant, raster: raster, immediates: immediates, drawEnabled: drawEnabled,
                               timestamps: timestamps, occlusion: occlusion) { encoder in
                    let topology = pipeline.primitive.topology.metal.type
                    if let (index, buffer, _) = indices {
                        encoder.drawIndexedPrimitives(type: topology, indexType: index.format.metal,
                                                      indexBuffer: buffer, indexBufferOffset: Int(index.offset),
                                                      indirectBuffer: validated!, indirectBufferOffset: 0)
                    } else {
                        encoder.drawPrimitives(type: topology, indirectBuffer: validated!, indirectBufferOffset: 0)
                    }
                }
            case .copyTextureToBuffer(let source, let destination, let bytesPerRow, let mipLevel, let aspect, let offset,
                                      let layer, let z, let depth, let rowsPerImage):
                if depth == 0 { continue }
                let texture = textures[source]!
                let extent = texture.descriptor.mipExtent(mipLevel)!
                let physical = texture.descriptor.mipPhysicalExtent(mipLevel)!
                let region = TextureCopyRegion(width: physical.width, height: physical.height, mipLevel: mipLevel,
                                               arrayLayer: layer, z: z, depth: depth)
                for plane in region.planes(in: texture.descriptor) {
                    try initializeTexture(source, level: mipLevel, layer: plane, on: commandBuffer, initialized: &initialized)
                }
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                let block = texture.descriptor.format.copyBlock(aspect: aspect)!
                let blockHeight = block.height
                let imageStride = depth > 1 ? Int(bytesPerRow) * Int(rowsPerImage ?? (physical.height / blockHeight)) : 0
                let metalRowStride = depth == 1 && physical.height == blockHeight
                    ? Int(block.rowBytes(width: physical.width)) : Int(bytesPerRow)
                let options: MTLBlitOption = switch aspect {
                case .stencilOnly where texture.descriptor.format.hasDepth: .stencilFromDepthStencil
                case .depthOnly where texture.descriptor.format.hasStencil: .depthFromDepthStencil
                default: []
                }
                let volume = texture.descriptor.dimension == .d3
                for slice in 0..<(volume ? 1 : Int(depth)) {
                    encoder.copy(from: texture.native, sourceSlice: Int(layer) + slice, sourceLevel: Int(mipLevel),
                                 sourceOrigin: MTLOrigin(x: 0, y: 0, z: Int(z)),
                                 sourceSize: MTLSize(width: Int(extent.width), height: Int(extent.height),
                                                     depth: volume ? Int(depth) : 1),
                                 to: buffers[destination]!.native, destinationOffset: Int(offset) + slice * imageStride,
                                 destinationBytesPerRow: metalRowStride,
                                 destinationBytesPerImage: volume && depth > 1 ? imageStride : 0,
                                 options: options)
                }
                encoder.endEncoding()
            case .copyTextureRegionToBuffer(let source, let destination, let layout, let region, let aspect):
                if region.isEmpty { continue }
                let texture = textures[source]!
                for plane in region.planes(in: texture.descriptor) {
                    try initializeTexture(source, level: region.mipLevel, layer: plane, on: commandBuffer, initialized: &initialized)
                }
                let block = texture.descriptor.format.copyBlock(aspect: aspect)!
                let stride = Int(layout.rowStride(width: region.width, block: block))
                let imageStride = region.depth > 1 ? stride * Int(layout.rowsPerImage ?? (region.height / block.height)) : 0
                let metalRowStride = region.depth == 1 && region.height == block.height
                    ? Int(block.rowBytes(width: region.width)) : stride
                let extent = texture.descriptor.mipExtent(region.mipLevel)!
                let volume = texture.descriptor.dimension == .d3
                let options: MTLBlitOption = switch aspect {
                case .stencilOnly where texture.descriptor.format.hasDepth: .stencilFromDepthStencil
                case .depthOnly where texture.descriptor.format.hasStencil: .depthFromDepthStencil
                default: []
                }
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                for slice in 0..<(volume ? 1 : Int(region.depth)) {
                    encoder.copy(from: texture.native, sourceSlice: Int(region.arrayLayer) + slice,
                                 sourceLevel: Int(region.mipLevel),
                                 sourceOrigin: MTLOrigin(x: Int(region.x), y: Int(region.y), z: Int(region.z)),
                                 sourceSize: MTLSize(width: Int(min(region.width, extent.width - region.x)),
                                                     height: Int(min(region.height, extent.height - region.y)),
                                                     depth: volume ? Int(region.depth) : 1),
                                 to: buffers[destination]!.native, destinationOffset: Int(layout.offset) + slice * imageStride,
                                 destinationBytesPerRow: metalRowStride,
                                 destinationBytesPerImage: volume && region.depth > 1 ? imageStride : 0, options: options)
                }
                encoder.endEncoding()
            case .copyBufferToTexture(let source, let layout, let destination, let region, let aspect):
                if region.isEmpty { continue }
                try encodeTextureUpload(buffers[source]!.native, layout: layout, destination: destination,
                                        region: region, aspect: aspect, on: commandBuffer, initialized: &initialized)
            case .writeTexture(let destination, let data, let layout, let region, let aspect):
                if region.isEmpty { continue }
                let block = textures[destination]!.descriptor.format.copyBlock(aspect: aspect)!
                let (staging, packedLayout) = try stageTextureUpload(data, layout: layout, region: region,
                                                                    block: block)
                try encodeTextureUpload(staging, layout: packedLayout, destination: destination, region: region,
                                        aspect: aspect, on: commandBuffer, initialized: &initialized)
                retained.append(staging)
            case .copyTextureToTexture(let source, let destination, let region, let origin):
                if region.isEmpty { continue }
                let sourceTexture = textures[source]!
                for plane in region.planes(in: sourceTexture.descriptor) {
                    try initializeTexture(source, level: region.mipLevel, layer: plane, on: commandBuffer, initialized: &initialized)
                }
                let target = textures[destination]!
                let destinationRegion = origin.region(width: region.width, height: region.height, depth: region.depth)
                try prepareTextureWrite(destination, region: destinationRegion, aspect: .all, on: commandBuffer, initialized: &initialized)
                let destinationTexture: any MTLTexture
                if sourceTexture.native.pixelFormat == target.native.pixelFormat {
                    destinationTexture = target.native
                } else {
                    guard let view = target.native.makeTextureView(pixelFormat: sourceTexture.native.pixelFormat) else {
                        throw GPUError.outOfMemory
                    }
                    destinationTexture = view
                    retainedCopyViews.append(view)
                }
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                let sourceVolume = sourceTexture.descriptor.dimension == .d3
                let destinationVolume = target.descriptor.dimension == .d3
                let size = MTLSize(width: Int(region.width), height: Int(region.height), depth: 1)
                for slice in 0..<Int(region.depth) {
                    encoder.copy(from: sourceTexture.native, sourceSlice: sourceVolume ? 0 : Int(region.arrayLayer) + slice,
                                 sourceLevel: Int(region.mipLevel),
                                 sourceOrigin: MTLOrigin(x: Int(region.x), y: Int(region.y), z: sourceVolume ? Int(region.z) + slice : 0), sourceSize: size,
                                 to: destinationTexture, destinationSlice: destinationVolume ? 0 : Int(origin.arrayLayer) + slice,
                                 destinationLevel: Int(origin.mipLevel),
                                 destinationOrigin: MTLOrigin(x: Int(origin.x), y: Int(origin.y), z: destinationVolume ? Int(origin.z) + slice : 0))
                }
                encoder.endEncoding()
            case .read(let id, let size):
                guard let snapshot = native.makeBuffer(length: Int(size), options: .storageModeShared),
                      let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.copy(from: buffers[id]!.native, sourceOffset: 0, to: snapshot,
                             destinationOffset: 0, size: Int(size))
                encoder.endEncoding()
                retained.append(snapshot)
                reads.append((snapshot, Int(size / 4)))
            }
        }
        var batch = Batch(
            commandBuffer: commandBuffer, buffers: retained, pipelines: retainedPipelines,
            textures: resources.compactMap { textures[$0]?.native ?? textureViews[$0]?.native } + retainedCopyViews,
            samplers: resources.compactMap { samplers[$0]?.native }, renderPipelines: retainedRenderPipelines, drawable: nil,
            meshPipelines: retainedMeshPipelines, counterSampleBuffers: resources.compactMap { querySets[$0]?.samples },
            committedHandlers: committedHandlers,
            accelerationBuilds: accelerationBuilds, retainedAccelerationBuilds: retainedAccelerationBuilds)
        if trace {
            stats.cpu = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) - cpuStart
            batch.groupingStats = stats
        }
        return (batch, reads)
    }
}
#endif
