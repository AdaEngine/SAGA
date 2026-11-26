import SAGABackend
import SAGAShaderIR
import SAGATypes
#if canImport(Metal)
import Metal
#endif
#if canImport(QuartzCore)
import QuartzCore
#endif
#if canImport(Darwin)
import Darwin
#endif

extension MetalDevice {
    public func submit(_ submission: Submission) async throws {
        _ = try await submitCommands(submission, commands: [])
    }

    public func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        try requireActive()
        #if canImport(Metal)
        // The device tail orders CPU uploads after earlier uses of these buffers.
        if submission.commandsValidated, inFlight.isEmpty,
           try applySharedUploads(commands, resources: submission.resources) {
            return []
        }

        let owner = submission.resources.first?.device ??
            commands.lazy.compactMap { $0.recordedPass.command.computeDispatch?.pipeline.id.device }.first ?? 0
        let pinned = Set(submission.resources)
        let descriptors = validationDescriptors(buffers, pinned: pinned, extract: \.descriptor)
        let queryDescriptors = validationDescriptors(querySets, pinned: pinned, extract: \.descriptor)
        let textureDescriptors = validationDescriptors(textures, pinned: pinned, extract: \.descriptor)
        let samplerDescriptors = validationDescriptors(samplers, pinned: pinned, extract: \.descriptor)
        let views = validationDescriptors(textureViews, pinned: pinned, extract: \.resource)
        let externalPlanes = validationDescriptors(externalTextures, pinned: pinned, extract: \.planes)
        let accelerationDescriptors = validationDescriptors(accelerationStructures, pinned: pinned, extract: \.descriptor)
        for id in submission.resources {
            guard buffers[id] != nil || textures[id] != nil || views[id] != nil || samplers[id] != nil ||
                  querySets[id] != nil || externalPlanes[id] != nil || accelerationStructures[id] != nil || meshPipelines[id] != nil else {
                throw GPUError.validation("Unknown Metal resource")
            }
            if let view = views[id], !pinned.contains(view.texture) {
                throw GPUError.validation("Unpinned texture view parent")
            }
            if let planes = externalPlanes[id], !planes.allSatisfy(pinned.contains) {
                throw GPUError.validation("Unpinned external texture plane")
            }
        }
        for recorded in commands {
            guard recorded.resources.allSatisfy(pinned.contains) else {
                throw GPUError.validation("Unpinned command resource")
            }
            try recorded.validate(device: owner, descriptors: descriptors, textures: textureDescriptors,
                                  samplerDescriptors: samplerDescriptors, textureViews: views,
                                  maximumViewportDimension: capabilities.maxTextureDimension2D, maxVertexBuffers: capabilities.maxVertexBuffers,
                                     maxBindGroupsPlusVertexBuffers: capabilities.maxBindGroupsPlusVertexBuffers, querySets: queryDescriptors,
                                  maxMultiviewViewCount: capabilities.maxMultiviewViewCount,
                                  features: capabilities.features, externalTextures: Set(externalPlanes.keys),
                                  externalPlanes: externalPlanes, accelerationStructures: accelerationDescriptors,
                                  maxComputeWorkgroupsPerDimension: capabilities.maxComputeWorkgroupsPerDimension,
                                  maxTaskWorkgroupTotalCount: capabilities.maxTaskWorkgroupTotalCount,
                                  maxTaskWorkgroupsPerDimension: capabilities.maxTaskWorkgroupsPerDimension,
                                  maxMeshWorkgroupTotalCount: capabilities.maxMeshWorkgroupTotalCount,
                                  maxMeshWorkgroupsPerDimension: capabilities.maxMeshWorkgroupsPerDimension)
            let command = recorded.recordedPass.command
            if case .dispatch(let id, _, _, _) = command, pipelines[id] == nil {
                throw GPUError.validation("Unknown Metal diagnostic pipeline")
            }
            if let dispatch = command.computeDispatch, computePipelines[dispatch.pipeline.id] == nil {
                throw GPUError.validation("Unknown Metal compute pipeline")
            }
            switch command {
            case .buildAccelerationStructure(let destination, let geometry):
                guard let target = accelerationStructures[destination], let buffer = buffers[geometry.vertexBuffer],
                      case .triangles(let maximum) = target.descriptor.capacity, geometry.triangleCount <= maximum,
                      Int(exactly: geometry.vertexOffset) != nil, Int(exactly: geometry.vertexStride) != nil else {
                    throw GPUError.validation("Invalid Metal acceleration structure build")
                }
                _ = try geometry.vertexRange(bufferSize: buffer.descriptor.size)
                if let index = geometry.indexBuffer {
                    guard let source = buffers[index.buffer], Int(exactly: index.offset) != nil else {
                        throw GPUError.validation("Invalid Metal acceleration structure index buffer")
                    }
                    _ = try geometry.indexRange(bufferSize: source.descriptor.size)
                }
            case .renderMultiIndirect:
                throw GPUError.unsupported("Metal does not run multi-draw indirect commands")
            case .buildBoxAccelerationStructure:
                throw GPUError.unsupported("Metal does not build box acceleration structures")
            case .buildTopLevelAccelerationStructure(let destination, let instances):
                guard let target = accelerationStructures[destination],
                      case .instances(let maximum) = target.descriptor.capacity,
                      instances.count <= Int(maximum), instances.allSatisfy({ instance in
                          guard let source = accelerationStructures[instance.accelerationStructure],
                                case .triangles = source.descriptor.capacity else { return false }
                          return true
                      }) else { throw GPUError.validation("Invalid Metal top-level acceleration structure build") }
            case .drawMesh(let pipeline, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _):
                guard meshPipelines[pipeline.id] != nil else { throw GPUError.validation("Unknown Metal mesh pipeline") }
            case .render(let pipeline, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
                 .renderIndexed(let pipeline, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
                 .renderIndirect(let pipeline, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _):
                guard renderPipelines[pipeline.id] != nil else {
                    throw GPUError.validation("Unknown Metal render pipeline")
                }
            default: break
            }
        }
        var initialized: [ResourceID: Set<TextureSubresource>] = [:]
        var writtenQueries: [ResourceID: Set<UInt32>] = [:]
        var encoded: [(batch: Batch, reads: [(any MTLBuffer, Int)],
                       initialized: [ResourceID: Set<TextureSubresource>], written: [ResourceID: Set<UInt32>])] = []
        var precedingAccelerationBuilds: [ResourceID: PendingAccelerationBuild] = [:]
        for range in queryCommandRanges(commands) {
            let (batch, reads) = try encodeCommands(commands[range], resources: submission.resources,
                                                    initialized: &initialized, writtenQueries: &writtenQueries,
                                                    precedingAccelerationBuilds: precedingAccelerationBuilds)
            precedingAccelerationBuilds.merge(batch.accelerationBuilds) { _, newer in newer }
            encoded.append((batch, reads, initialized, writtenQueries))
        }
        var snapshots: [[UInt32]] = []
        for item in encoded {
            if let stats = item.batch.groupingStats {
                print("SAGA-ENC\t\(submission.sequence)\t\(stats.cpu)\t\(stats.indexedDraws)\t\(stats.renderEncoders)\t\(stats.indexEncoders)\t\(stats.groups)\t\(stats.groupedDraws)")
            }
            try await commit(item.batch, initialized: item.initialized, writtenQueries: item.written,
                             traceSequence: Self.traceGrouping ? submission.sequence : nil)
            snapshots += item.reads.map { buffer, count in
                Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: count))
            }
        }
        return snapshots
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    #if canImport(Metal)
    private func validationDescriptors<Value, Descriptor>(
        _ all: [ResourceID: Value],
        pinned: Set<ResourceID>,
        extract: (Value) -> Descriptor
    ) -> [ResourceID: Descriptor] {
        var result: [ResourceID: Descriptor] = [:]
        for id in pinned {
            if let resource = all[id] { result[id] = extract(resource) }
        }
        return result
    }

    /// Metal can publish pass counter samples after subsequent encoders have started.
    /// End the command buffer before another command consumes or overwrites those samples.
    private func queryCommandRanges(_ commands: [BufferCommand]) -> [Range<Int>] {
        var ranges: [Range<Int>] = []
        var start = 0
        var written: [ResourceID: Set<UInt32>] = [:]
        var resolved: Set<ResourceID> = []
        func splitBefore(_ index: Int) {
            ranges.append(start..<index)
            start = index
            written.removeAll()
            resolved.removeAll()
        }
        for (index, command) in commands.enumerated() {
            switch command {
            case .timestamped, .occlusionQueried:
                let (_, timestamps, occlusion) = command.recordedPass
                let samples = [timestamps.map { ($0.querySet, $0.indices) },
                               occlusion.map { ($0.querySet, [$0.queryIndex]) }].compactMap { $0 }
                if samples.contains(where: { id, indices in
                    resolved.contains(id) || !written[id, default: []].isDisjoint(with: indices)
                }) {
                    splitBefore(index)
                }
                for (id, indices) in samples { written[id, default: []].formUnion(indices) }
            case .resolveQuerySet(let id, _, let count, _, _) where count > 0:
                if written[id] != nil { splitBefore(index) }
                resolved.insert(id)
            default:
                break
            }
        }
        ranges.append(start..<commands.count)
        return ranges
    }

    func commit(_ batch: Batch, initialized: [ResourceID: Set<TextureSubresource>],
                        writtenQueries: [ResourceID: Set<UInt32>] = [:], traceSequence: UInt64? = nil) async throws {
        guard nextBatch < UInt64.max else { throw GPUError.outOfMemory }
        nextBatch += 1
        let serial = nextBatch
        let completion = BackendCompletion()
        inFlight[serial] = batch
        let retirementHook = retirementHook
        batch.commandBuffer.addCompletedHandler { [self] buffer in
            if let traceSequence, buffer.gpuEndTime >= buffer.gpuStartTime, buffer.gpuStartTime > 0 {
                print("SAGA-GPU\t\(traceSequence)\t\(buffer.gpuStartTime)\t\(buffer.gpuEndTime)")
            }
            let result: Result<Void, GPUError> = buffer.status == .completed ? .success(()) :
                .failure(.deviceLost("Metal command failed: \(String(describing: buffer.error))"))
            Task {
                if let retirementHook { await retirementHook() }
                await retire(serial, completion: completion, result: result)
            }
        }
        let commitCPU = traceSequence != nil ? clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) : 0
        batch.commandBuffer.commit()
        if let traceSequence {
            print("SAGA-COMMIT\t\(traceSequence)\t\(clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) - commitCPU)")
        }
        for handler in batch.committedHandlers { handler() }
        for (id, build) in batch.accelerationBuilds {
            if let current = accelerationStructures[id], (current.build?.serial ?? 0) < build.serial {
                accelerationStructures[id]?.build = build
            }
        }
        for (id, levels) in initialized { textures[id]?.initializedSubresources.formUnion(levels) }
        for (id, indices) in writtenQueries { querySets[id]?.written.formUnion(indices) }
        do {
            try await completion.wait(timeout: timeout)
        }
        catch { terminal = true; throw error }
    }

    /// An empty Metal encoder can be skipped, including its counter attachments.
    func encodeTimestampMarker(
        _ commands: any MTLCommandBuffer, writes: PassTimestampWrites
    ) throws -> any MTLBuffer {
        let encoder = try computeEncoder(commands, timestamps: writes)
        defer { encoder.endEncoding() }
        return try encodeTimestampWork(encoder)
    }

    func encodeTimestampWork(_ encoder: any MTLComputeCommandEncoder) throws -> any MTLBuffer {
        if timestampMarkerPipeline == nil {
            let library = try native.makeLibrary(source: """
                #include <metal_stdlib>
                using namespace metal;
                kernel void saga_timestamp_marker(device uint& scratch [[buffer(0)]]) { scratch = 1u; }
                """, options: nil)
            guard let function = library.makeFunction(name: "saga_timestamp_marker") else {
                throw GPUError.backendFailure("Metal timestamp marker entry point missing")
            }
            timestampMarkerPipeline = try native.makeComputePipelineState(function: function)
        }
        guard let scratch = native.makeBuffer(length: 4, options: .storageModePrivate) else { throw GPUError.outOfMemory }
        encoder.setComputePipelineState(timestampMarkerPipeline!)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        let one = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(one, threadsPerThreadgroup: one)
        return scratch
    }

    func resolveQueries(
        _ query: QuerySet, written: Set<UInt32>, range: Range<UInt32>, destination: any MTLBuffer,
        offset: Int, on commands: any MTLCommandBuffer
    ) throws {
        guard let encoder = commands.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
        var first = range.lowerBound
        var destinationOffset = offset
        while first < range.upperBound {
            let hasSample = written.contains(first)
            var end = first + 1
            while end < range.upperBound && written.contains(end) == hasSample { end += 1 }
            let length = Int(end - first) * 8
            if hasSample {
                switch query.storage {
                case .timestamp(let samples):
                    encoder.resolveCounters(samples, range: Int(first)..<Int(end),
                                            destinationBuffer: destination, destinationOffset: destinationOffset)
                case .occlusion(let visibility):
                    encoder.copy(from: visibility, sourceOffset: Int(first) * 8, to: destination,
                                 destinationOffset: destinationOffset, size: length)
                }
            } else {
                encoder.fill(buffer: destination, range: destinationOffset..<(destinationOffset + length), value: 0)
            }
            destinationOffset += length
            first = end
        }
        encoder.endEncoding()
    }

    private func retire(_ serial: UInt64, completion: BackendCompletion, result: Result<Void, GPUError>) async {
        inFlight.removeValue(forKey: serial)
        if case .failure = result { terminal = true }
        await completion.finish(result)
        if inFlight.isEmpty {
            let waiters = retirementWaiters; retirementWaiters.removeAll()
            for waiter in waiters { waiter.resume() }
        }
    }

    // Internal deterministic fault-injection seam, never used by the production diagnostic.
    func setRetirementHookForTesting(_ hook: @escaping @Sendable () async -> Void) { retirementHook = hook }
    func resourceCountsForTesting() -> (buffers: Int, pipelines: Int, batches: Int, retainedBuffers: Int) {
        (buffers.count, pipelines.count + computePipelines.count, inFlight.count,
         inFlight.values.reduce(0) { $0 + $1.buffers.count })
    }
    func waitForRetirementForTesting() async {
        if !inFlight.isEmpty { await withCheckedContinuation { retirementWaiters.append($0) } }
    }
    #endif
}
