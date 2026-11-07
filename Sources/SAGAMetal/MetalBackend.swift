import SAGATypes
import SAGABackend
import SAGAShaderIR
#if canImport(Metal)
import Metal
#endif

public struct MetalAdapter: Sendable {
    public let name: String
    #if canImport(Metal)
    fileprivate let native: any MTLDevice
    fileprivate init(_ native: any MTLDevice) { self.native = native; name = native.name }
    #endif
}

public enum MetalBackend {
    public static func requestAdapter() -> MetalAdapter? {
        #if canImport(Metal)
        MTLCreateSystemDefaultDevice().map(MetalAdapter.init)
        #else
        nil
        #endif
    }

    public static func makeDevice(timeout: Duration = .seconds(10)) throws -> MetalDevice {
        try makeDevice(adapter: requestAdapter(), timeout: timeout)
    }

    public static func makeDevice(adapter: MetalAdapter?, timeout: Duration = .seconds(10)) throws -> MetalDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        return try MetalDevice(adapter: adapter, timeout: timeout)
    }
}

public actor MetalDevice: ComputeBackendDevice, RenderBackendDevice {
    public nonisolated let capabilities: DeviceCapabilities
    public nonisolated let adapterName: String
    private let timeout: Duration
    private var terminal = false
    #if canImport(Metal)
    private let native: any MTLDevice
    private let queue: any MTLCommandQueue
    private struct Buffer {
        let descriptor: BufferDescriptor
        let native: any MTLBuffer
    }
    private struct Batch {
        let commandBuffer: any MTLCommandBuffer
        let buffers: [any MTLBuffer]
        let pipelines: [any MTLComputePipelineState]
        let textures: [any MTLTexture]
        let renderPipelines: [any MTLRenderPipelineState]
    }
    private struct Texture {
        let descriptor: TextureDescriptor
        let native: any MTLTexture
        var initialized = false
    }
    private struct Pipeline {
        let state: any MTLComputePipelineState
        let threadsPerThreadgroup: MTLSize
    }
    private var buffers: [ResourceID: Buffer] = [:]
    private var pipelines: [ResourceID: any MTLComputePipelineState] = [:]
    private var computePipelines: [ResourceID: Pipeline] = [:]
    private var textures: [ResourceID: Texture] = [:]
    private var renderPipelines: [ResourceID: any MTLRenderPipelineState] = [:]
    private var cachedDiagnosticPipeline: (any MTLComputePipelineState)?
    private var retirementHook: (@Sendable () async -> Void)?
    private var retirementWaiters: [CheckedContinuation<Void, Never>] = []
    private var nextPipeline: UInt64 = 0
    private var nextBatch: UInt64 = 0
    private var inFlight: [UInt64: Batch] = [:]
    #endif

    fileprivate init(adapter: MetalAdapter, timeout: Duration) throws {
        guard timeout > .zero else { throw GPUError.validation("Timeout must be positive") }
        self.timeout = timeout
        adapterName = adapter.name
        #if canImport(Metal)
        native = adapter.native
        guard let queue = adapter.native.makeCommandQueue() else { throw GPUError.outOfMemory }
        self.queue = queue
        capabilities = try DeviceCapabilities(maxBufferSize: UInt64(adapter.native.maxBufferLength))
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    private func requireActive() throws {
        guard !terminal else { throw GPUError.deviceLost("Metal device is terminal after command failure/timeout") }
    }

    public func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        guard !descriptor.mappedAtCreation else { throw GPUError.unsupported("Mapped-at-creation API not implemented") }
        #if canImport(Metal)
        guard buffers[id] == nil, textures[id] == nil, let length = Int(exactly: descriptor.size) else {
            throw GPUError.validation("Duplicate buffer or unrepresentable size")
        }
        guard let buffer = native.makeBuffer(length: max(1, length), options: .storageModeShared) else {
            throw GPUError.outOfMemory
        }
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: max(1, length))
        buffer.label = descriptor.label
        buffers[id] = Buffer(descriptor: descriptor, native: buffer)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyBuffer(id: ResourceID) {
        #if canImport(Metal)
        buffers.removeValue(forKey: id)
        #endif
    }

    public func close(owner: UInt64) {
        #if canImport(Metal)
        buffers = buffers.filter { $0.key.device != owner }
        textures = textures.filter { $0.key.device != owner }
        pipelines = pipelines.filter { $0.key.device != owner }
        computePipelines = computePipelines.filter { $0.key.device != owner }
        renderPipelines = renderPipelines.filter { $0.key.device != owner }
        #endif
    }

    public func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) throws {
        try requireActive()
        #if canImport(Metal)
        guard case .text(.msl, let source) = shader.artifact else {
            throw GPUError.validation("Metal requires a compiled MSL compute shader")
        }
        guard computePipelines[pipeline.id] == nil, shader.storageBindings.allSatisfy({
            $0.binding.group == 0 && $0.binding.binding < 8
        }) else { throw GPUError.validation("Invalid Metal pipeline identity or binding layout") }

        let size = shader.workgroupSize
        let threads = MTLSize(width: Int(size.x), height: Int(size.y), depth: Int(size.z))
        let limit = native.maxThreadsPerThreadgroup
        guard (1...limit.width).contains(threads.width),
              (1...limit.height).contains(threads.height),
              (1...limit.depth).contains(threads.depth)
        else { throw GPUError.validation("Workgroup dimensions exceed Metal device limits") }
        let state: any MTLComputePipelineState
        do {
            let library = try native.makeLibrary(source: source, options: nil)
            guard let function = library.makeFunction(name: shader.entryPoint) else {
                throw GPUError.backendFailure("Compiled Metal entry point missing")
            }
            state = try native.makeComputePipelineState(function: function)
        } catch let error as GPUError {
            throw error
        } catch {
            throw GPUError.backendFailure("Compute MSL compilation: \(error)")
        }
        guard threads.width * threads.height * threads.depth <= state.maxTotalThreadsPerThreadgroup else {
            throw GPUError.validation("Workgroup size exceeds Metal pipeline limits")
        }
        computePipelines[pipeline.id] = Pipeline(state: state, threadsPerThreadgroup: threads)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyComputePipeline(id: ResourceID) {
        #if canImport(Metal)
        computePipelines.removeValue(forKey: id)
        #endif
    }

    public func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        #if canImport(Metal)
        guard textures[id] == nil, buffers[id] == nil else { throw GPUError.validation("Duplicate Metal resource") }
        let metal = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: descriptor.format.metal, width: Int(descriptor.width),
            height: Int(descriptor.height), mipmapped: false)
        metal.storageMode = .private
        metal.usage = .renderTarget
        guard let texture = native.makeTexture(descriptor: metal) else { throw GPUError.outOfMemory }
        textures[id] = Texture(descriptor: descriptor, native: texture)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyTexture(id: ResourceID) {
        #if canImport(Metal)
        textures.removeValue(forKey: id)
        #endif
    }

    public func allocateRenderPipeline(
        _ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact
    ) throws {
        try requireActive()
        #if canImport(Metal)
        guard renderPipelines[pipeline.id] == nil, vertex.stage == .vertex, fragment.stage == .fragment,
              case .text(.msl, let vertexSource) = vertex.artifact,
              case .text(.msl, let fragmentSource) = fragment.artifact else {
            throw GPUError.validation("Metal requires compiled MSL vertex and fragment stages")
        }
        do {
            let vs = try native.makeLibrary(source: vertexSource, options: nil)
            let fs = try native.makeLibrary(source: fragmentSource, options: nil)
            guard let vertexFunction = vs.makeFunction(name: vertex.entryPoint),
                  let fragmentFunction = fs.makeFunction(name: fragment.entryPoint) else {
                throw GPUError.backendFailure("Compiled Metal render entry point missing")
            }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = vertexFunction
            descriptor.fragmentFunction = fragmentFunction
            descriptor.colorAttachments[0].pixelFormat = pipeline.colorFormat.metal
            if !pipeline.vertexBuffers.isEmpty {
                let vertexDescriptor = MTLVertexDescriptor()
                for (slot, layout) in pipeline.vertexBuffers.enumerated() {
                    let buffer = vertexDescriptor.layouts[slot]!
                    buffer.stride = layout.stride == 0 ? Int(layout.attributeEnd) : Int(layout.stride)
                    buffer.stepFunction = layout.stride == 0 ? .constant : .perVertex
                    buffer.stepRate = layout.stride == 0 ? 0 : 1
                    for attribute in layout.attributes {
                        let destination = vertexDescriptor.attributes[Int(attribute.location)]!
                        destination.format = attribute.format.metal
                        destination.bufferIndex = slot
                        destination.offset = Int(attribute.offset)
                    }
                }
                descriptor.vertexDescriptor = vertexDescriptor
            }
            renderPipelines[pipeline.id] = try native.makeRenderPipelineState(descriptor: descriptor)
        } catch let error as GPUError {
            throw error
        } catch {
            throw GPUError.backendFailure("Render MSL compilation: \(error)")
        }
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyRenderPipeline(id: ResourceID) {
        #if canImport(Metal)
        renderPipelines.removeValue(forKey: id)
        #endif
    }

    public func makeDiagnosticPipeline(owner: UInt64) throws -> ResourceID {
        try requireActive()
        #if canImport(Metal)
        guard owner > 0, nextPipeline < UInt64.max else { throw GPUError.validation("Invalid pipeline owner/serial") }
        if let existing = pipelines.keys.first(where: { $0.device == owner }) { return existing }
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void saga_diagnostic(device const uint* input [[buffer(0)]],
                                    device uint* output [[buffer(1)]],
                                    constant uint& count [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
            if (i < count) { output[i] = input[i] * 3u + 7u; }
        }
        """
        do {
            let pipeline: any MTLComputePipelineState
            if let cachedDiagnosticPipeline { pipeline = cachedDiagnosticPipeline }
            else {
                let library = try native.makeLibrary(source: source, options: nil)
                guard let function = library.makeFunction(name: "saga_diagnostic") else {
                    throw GPUError.backendFailure("Diagnostic entry point missing")
                }
                pipeline = try native.makeComputePipelineState(function: function)
                cachedDiagnosticPipeline = pipeline
            }
            nextPipeline += 1
            let id = ResourceID(device: owner, serial: nextPipeline)
            pipelines[id] = pipeline
            return id
        } catch { throw GPUError.backendFailure("Diagnostic MSL compilation: \(error)") }
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func submit(_ submission: Submission) async throws {
        _ = try await submitCommands(submission, commands: [])
    }

    public func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        try requireActive()
        #if canImport(Metal)
        let owner = submission.resources.first?.device ?? commands.lazy.compactMap { command -> UInt64? in
            if case .dispatchCompute(let pipeline, _, _) = command { return pipeline.id.device }
            return nil
        }.first ?? 0
        let descriptors = buffers.mapValues(\.descriptor)
        let textureDescriptors = textures.mapValues(\.descriptor)
        for id in submission.resources {
            guard buffers[id] != nil || textures[id] != nil else { throw GPUError.validation("Unknown Metal resource") }
        }
        for command in commands {
            guard command.resources.allSatisfy({ submission.resources.contains($0) }) else {
                throw GPUError.validation("Unpinned command resource")
            }
            try command.validate(device: owner, descriptors: descriptors, textures: textureDescriptors)
            if case .dispatch(let id, _, _, _) = command, pipelines[id] == nil {
                throw GPUError.validation("Unknown Metal diagnostic pipeline")
            }
            if case .dispatchCompute(let pipeline, _, _) = command, computePipelines[pipeline.id] == nil {
                throw GPUError.validation("Unknown Metal compute pipeline")
            }
            if case .render(let pipeline, _, _, _, _) = command, renderPipelines[pipeline.id] == nil {
                throw GPUError.validation("Unknown Metal render pipeline")
            }
        }
        guard let commandBuffer = queue.makeCommandBuffer() else { throw GPUError.outOfMemory }
        var retained = submission.resources.compactMap { buffers[$0]?.native }
        var retainedPipelines: [any MTLComputePipelineState] = []
        var retainedRenderPipelines: [any MTLRenderPipelineState] = []
        var initialized: Set<ResourceID> = []
        var reads: [(any MTLBuffer, Int)] = []
        for command in commands {
            switch command {
            case .write(let id, let offset, let words):
                let length = words.count * 4
                guard let staging = native.makeBuffer(length: length, options: .storageModeShared),
                      let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                words.withUnsafeBytes { staging.contents().copyMemory(from: $0.baseAddress!, byteCount: length) }
                encoder.copy(from: staging, sourceOffset: 0, to: buffers[id]!.native,
                             destinationOffset: Int(offset), size: length)
                encoder.endEncoding()
                retained.append(staging)
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
            case .dispatchCompute(let pipeline, let bindings, let workgroups):
                if workgroups.isEmpty { continue }
                let compute = computePipelines[pipeline.id]!
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.setComputePipelineState(compute.state)
                for (slot, buffer) in bindings {
                    encoder.setBuffer(buffers[buffer]!.native, offset: 0, index: Int(slot.binding))
                }
                encoder.dispatchThreadgroups(
                    MTLSize(width: Int(workgroups.x), height: Int(workgroups.y), depth: Int(workgroups.z)),
                    threadsPerThreadgroup: compute.threadsPerThreadgroup)
                encoder.endEncoding()
                retainedPipelines.append(compute.state)
            case .copy(let source, let destination, let size):
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.copy(from: buffers[source]!.native, sourceOffset: 0,
                             to: buffers[destination]!.native, destinationOffset: 0, size: Int(size))
                encoder.endEncoding()
            case .render(let pipeline, let target, let clear, let vertexCount, let vertexBuffers):
                let state = renderPipelines[pipeline.id]!
                let encoder = try renderEncoder(
                    commandBuffer, target: textures[target]!.native,
                    clear: MTLClearColor(red: clear.red, green: clear.green, blue: clear.blue, alpha: clear.alpha))
                if vertexCount > 0 {
                    encoder.setRenderPipelineState(state)
                    for (slot, binding) in vertexBuffers.enumerated() {
                        encoder.setVertexBuffer(buffers[binding.buffer]!.native, offset: Int(binding.offset), index: slot)
                    }
                    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: Int(vertexCount))
                }
                encoder.endEncoding()
                retainedRenderPipelines.append(state)
                initialized.insert(target)
            case .copyTextureToBuffer(let source, let destination, let bytesPerRow):
                let texture = textures[source]!
                if !texture.initialized && initialized.insert(source).inserted {
                    // Private Metal allocations need a clear before their first observable read.
                    let clear = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
                    try renderEncoder(commandBuffer, target: texture.native, clear: clear).endEncoding()
                }
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.copy(from: texture.native, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                             sourceSize: MTLSize(width: texture.native.width, height: texture.native.height, depth: 1),
                             to: buffers[destination]!.native, destinationOffset: 0,
                             destinationBytesPerRow: Int(bytesPerRow), destinationBytesPerImage: 0)
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
        guard nextBatch < UInt64.max else { throw GPUError.outOfMemory }
        nextBatch += 1
        let serial = nextBatch
        let completion = BackendCompletion()
        inFlight[serial] = Batch(
            commandBuffer: commandBuffer, buffers: retained, pipelines: retainedPipelines,
            textures: submission.resources.compactMap { textures[$0]?.native },
            renderPipelines: retainedRenderPipelines)
        let retirementHook = retirementHook
        commandBuffer.addCompletedHandler { [self] buffer in
            let result: Result<Void, GPUError> = buffer.status == .completed ? .success(()) :
                .failure(.deviceLost("Metal command failed: \(String(describing: buffer.error))"))
            Task {
                if let retirementHook { await retirementHook() }
                await retire(serial, completion: completion, result: result)
            }
        }
        commandBuffer.commit()
        for id in initialized { textures[id]?.initialized = true }
        do { try await completion.wait(timeout: timeout) }
        catch { terminal = true; throw error }
        return reads.map { buffer, count in
            Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: count))
        }
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    #if canImport(Metal)
    private func renderEncoder(
        _ commands: any MTLCommandBuffer, target: any MTLTexture, clear: MTLClearColor
    ) throws -> any MTLRenderCommandEncoder {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = clear
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { throw GPUError.outOfMemory }
        return encoder
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

#if canImport(Metal)
private extension VertexFormat {
    var metal: MTLVertexFormat {
        switch self {
        case .float32: .float
        case .float32x2: .float2
        case .float32x3: .float3
        case .float32x4: .float4
        case .uint32: .uint
        case .uint32x2: .uint2
        case .uint32x3: .uint3
        case .uint32x4: .uint4
        case .sint32: .int
        case .sint32x2: .int2
        case .sint32x3: .int3
        case .sint32x4: .int4
        }
    }
}

private extension TextureFormat {
    var metal: MTLPixelFormat {
        switch self { case .rgba8Unorm: .rgba8Unorm; case .bgra8Unorm: .bgra8Unorm }
    }
}
#endif
