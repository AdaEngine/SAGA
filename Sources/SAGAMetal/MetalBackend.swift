import SAGATypes
import SAGABackend
import SAGAShaderIR
#if canImport(Metal)
import Metal
#endif
#if canImport(QuartzCore)
import QuartzCore
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
        return try MetalDevice(adapter: adapter, timeout: timeout, surface: nil)
    }

    #if canImport(Metal) && canImport(QuartzCore)
    @MainActor
    public static func makeDevice(
        surface: MetalPresentationSurface, adapter: MetalAdapter? = requestAdapter(),
        timeout: Duration = .seconds(10)
    ) throws -> MetalDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        let backend = try MetalDevice(adapter: adapter, timeout: timeout, surface: surface)
        try surface.attach(to: adapter.native)
        return backend
    }
    #endif
}

@MainActor
public final class MetalDevice: ComputeBackendDevice, PresentationBackendDevice, SamplerBackendDevice, TextureViewBackendDevice,
                                QuerySetBackendDevice, BufferMappingBackendDevice, PlanarExternalTextureBackendDevice {
    public nonisolated let capabilities: DeviceCapabilities
    public nonisolated let adapterName: String
    public nonisolated let supportsPresentation: Bool
    private let timeout: Duration
    private var terminal = false
    #if canImport(Metal)
    private let native: any MTLDevice
    private let queue: any MTLCommandQueue
    private let surface: MetalPresentationSurface?
    private let timestampCounters: (any MTLCounterSet)?
    private struct Buffer {
        let descriptor: BufferDescriptor
        let native: any MTLBuffer
        var mapping: BufferMapping?
    }
    private struct QuerySet {
        enum Storage {
            case timestamp(any MTLCounterSampleBuffer)
            case occlusion(any MTLBuffer)
        }
        let descriptor: QuerySetDescriptor
        let storage: Storage
        var written: Set<UInt32> = []

        var samples: (any MTLCounterSampleBuffer)? {
            if case .timestamp(let samples) = storage { return samples }
            return nil
        }

        var visibility: (any MTLBuffer)? {
            if case .occlusion(let visibility) = storage { return visibility }
            return nil
        }
    }
    private struct Batch {
        let commandBuffer: any MTLCommandBuffer
        let buffers: [any MTLBuffer]
        let pipelines: [any MTLComputePipelineState]
        let textures: [any MTLTexture]
        let samplers: [any MTLSamplerState]
        let renderPipelines: [RenderState]
        let drawable: (any MTLDrawable)?
        var counterSampleBuffers: [any MTLCounterSampleBuffer] = []
    }
    private struct RenderState {
        let pipeline: any MTLRenderPipelineState
        let depth: (any MTLDepthStencilState)?
        let writesDepth: Bool
        let vertexRuntimeLengths: [(ComputeBindingSlot, UInt32, UInt32)]
        let fragmentRuntimeLengths: [(ComputeBindingSlot, UInt32, UInt32)]
        let vertexExternalTextures: [ComputeShaderArtifact.ExternalTextureLayout]
        let fragmentExternalTextures: [ComputeShaderArtifact.ExternalTextureLayout]
        let fragmentTextureAtomicGuardBufferIndex: UInt32?
        let vertexImmediateBufferIndex: UInt32?
        let fragmentImmediateBufferIndex: UInt32?
    }

    private struct TextureSubresource: Hashable {
        let level: UInt32
        let layer: UInt32
    }

    private struct Texture {
        let descriptor: TextureDescriptor
        let native: any MTLTexture
        var initializedSubresources: Set<TextureSubresource> = []
    }
    private struct Sampler {
        let native: any MTLSamplerState
        let descriptor: SamplerDescriptor
    }

    private struct TextureView {
        let resource: TextureViewResource
        let native: any MTLTexture
    }
    private struct ExternalTexture {
        let planes: [ResourceID]
        let parameters: [UInt32]
    }
    private struct Pipeline {
        let state: any MTLComputePipelineState
        let threadsPerThreadgroup: MTLSize
        let bufferIndices: [ComputeBindingSlot: Int]
        let textureIndices: [ComputeBindingSlot: Int]
        let samplerIndices: [ComputeBindingSlot: Int]
        let runtimeLengths: [(slot: ComputeBindingSlot, stride: UInt32, offset: UInt32)]
        let lengthBufferIndex: UInt32?
        let immediateBufferIndex: UInt32?
        let externalTextures: [ComputeShaderArtifact.ExternalTextureLayout]
    }
    private var buffers: [ResourceID: Buffer] = [:]
    private var querySets: [ResourceID: QuerySet] = [:]
    private var pipelines: [ResourceID: any MTLComputePipelineState] = [:]
    private var computePipelines: [ResourceID: Pipeline] = [:]
    private var textures: [ResourceID: Texture] = [:]
    private var samplers: [ResourceID: Sampler] = [:]
    private var textureViews: [ResourceID: TextureView] = [:]
    private var externalTextures: [ResourceID: ExternalTexture] = [:]
    private var renderPipelines: [ResourceID: RenderState] = [:]
    private var surfacePipeline: (any MTLRenderPipelineState)?
    private var presenting = false
    private var indexValidator: MetalIndexValidator?
    private var dispatchValidator: MetalDispatchValidator?
    private var indirectDrawValidator: MetalIndirectDrawValidator?
    private var timestampMarkerPipeline: (any MTLComputePipelineState)?
    private var cachedDiagnosticPipeline: (any MTLComputePipelineState)?
    private var retirementHook: (@Sendable () async -> Void)?
    private var retirementWaiters: [CheckedContinuation<Void, Never>] = []
    private var nextPipeline: UInt64 = 0
    private var nextBatch: UInt64 = 0
    private var inFlight: [UInt64: Batch] = [:]
    #endif

    nonisolated fileprivate init(adapter: MetalAdapter, timeout: Duration, surface: MetalPresentationSurface?) throws {
        guard timeout > .zero else { throw GPUError.validation("Timeout must be positive") }
        self.timeout = timeout
        adapterName = adapter.name
        supportsPresentation = surface != nil
        #if canImport(Metal)
        native = adapter.native
        self.surface = surface
        guard let queue = adapter.native.makeCommandQueue() else { throw GPUError.outOfMemory }
        self.queue = queue
        var features: Set<GPUFeature> = [.storageTextureRead, .shaderF16, .indirectFirstInstance, .clipDistances,
                                         .shaderEarlyDepthTest, .linearIndexing, .subgroupUniformity]
        let counters = adapter.native.supportsCounterSampling(.atStageBoundary) ?
            adapter.native.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) : nil
        self.timestampCounters = counters
        if counters != nil { features.insert(.timestampQuery) }
        #if os(macOS)
        features.insert(.depthClipControl)
        features.insert(.dualSourceBlending)
        features.insert(.float32Blendable)
        features.insert(.normalized16Resolve)
        features.insert(.snorm8Renderable)
        features.formUnion([.packedColorStorageTextures, .rg11b10UfloatRenderable])
        if adapter.native.supportsFamily(.apple3) { features.insert(.rgb9e5Renderable) }
        if adapter.native.supportsBCTextureCompression {
            features.formUnion([.textureCompressionBC, .textureCompressionBCSliced3D])
        }
        if adapter.native.supportsFamily(.apple7) { features.insert(.textureCompressionETC2) }
        #else
        if adapter.native.supportsFamily(.apple4) { features.insert(.depthClipControl) }
        if adapter.native.supportsFamily(.apple1) { features.insert(.dualSourceBlending) }
        features.formUnion([.textureCompressionETC2, .rgb9e5Renderable])
        if adapter.native.supportsFamily(.apple2) { features.insert(.snorm8Renderable) }
        if adapter.native.supportsFamily(.apple3) {
            features.formUnion([.packedColorStorageTextures, .rg11b10UfloatRenderable])
        }
        #endif
        features.formUnion([.depth32FloatStencil8, .bgra8UnormStorage])
        if #available(macOS 14, iOS 17, *) { features.insert(.textureAtomic) }
        if #available(macOS 14, iOS 17, *), adapter.native.supportsFamily(.apple9) {
            features.insert(.textureInt64Atomic)
        }
        if adapter.native.supportsFamily(.metal3) || adapter.native.supportsFamily(.apple3) {
            features.insert(.shaderInt64)
        }
        if adapter.native.supportsFamily(.apple9) ||
            (adapter.native.supportsFamily(.apple8) && adapter.native.supportsFamily(.mac2)) {
            features.insert(.shaderInt64AtomicMinMax)
        }
        if adapter.native.supports32BitFloatFiltering { features.insert(.float32Filterable) }
        if adapter.native.supportsFamily(.apple2) { features.insert(.textureCompressionASTC) }
        if adapter.native.supportsFamily(.apple3) { features.insert(.textureCompressionASTCSliced3D) }
        if adapter.native.supportsFamily(.apple6) { features.insert(.textureCompressionASTCHDR) }
        features.formUnion([.halfFloatStorageTextures, .narrowInteger16StorageTextures, .normalized16Textures,
                            .normalized16Filterable, .narrow8BitStorageTextures])
        if adapter.native.readWriteTextureSupport != .tierNone { features.insert(.storageTextureReadWrite) }
        if adapter.native.readWriteTextureSupport == .tier2 {
            features.formUnion([.r16FloatStorageReadWrite, .integer16StorageReadWrite, .storage8BitReadWrite, .wideRGBAStorageReadWrite])
        }
        if adapter.native.supportsFamily(.metal3) || adapter.native.supportsFamily(.apple7) || adapter.native.supportsFamily(.mac2) {
            features.insert(.primitiveIndex)
            features.insert(.subgroups)
            features.insert(.subgroupSizeControl)
            features.insert(.subgroupID)
        }
        if adapter.native.supportsShaderBarycentricCoordinates { features.insert(.shaderBarycentrics) }
        // Intel Metal timestamps use a 12 MHz counter in the pinned wgpu profile.
        let period = adapter.name.hasPrefix("Intel") ? 1_000.0 / 12 : 1
        var float32MSAA: Set<TextureFormat> = []
        if adapter.native.supportsTextureSampleCount(4) {
            float32MSAA.formUnion([.r32Float, .rg32Float])
            #if os(macOS)
            float32MSAA.insert(.rgba32Float)
            #else
            if adapter.native.supportsFamily(.apple7) { float32MSAA.insert(.rgba32Float) }
            #endif
        }
        capabilities = try DeviceCapabilities(maxBufferSize: UInt64(adapter.native.maxBufferLength), features: features,
                                             timestampPeriodNanoseconds: period, multisampledFloat32Formats: float32MSAA,
                                             maxImmediateSize: 4096,
                                             maxComputeWorkgroupStorageSize: UInt32(clamping: adapter.native.maxThreadgroupMemoryLength))
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
        #if canImport(Metal)
        guard buffers[id] == nil, textures[id] == nil, samplers[id] == nil, textureViews[id] == nil, let length = Int(exactly: descriptor.size) else {
            throw GPUError.validation("Duplicate buffer or unrepresentable size")
        }
        guard let buffer = native.makeBuffer(length: max(1, length), options: .storageModeShared) else {
            throw GPUError.outOfMemory
        }
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: max(1, length))
        buffer.label = descriptor.label
        let mapping = descriptor.mappedAtCreation ?
            BufferMapping(buffer: id, mode: .write, offset: 0, size: descriptor.size, generation: 0) : nil
        buffers[id] = Buffer(descriptor: descriptor, native: buffer, mapping: mapping)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyBuffer(id: ResourceID) {
        #if canImport(Metal)
        buffers.removeValue(forKey: id)
        #endif
    }

    public func beginBufferMapping(id: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64) throws {
        try requireActive()
        #if canImport(Metal)
        let usage: BufferUsage = mode == .read ? .mapRead : .mapWrite
        guard let buffer = buffers[id], buffer.mapping == nil, buffer.descriptor.usage.contains(usage),
              offset % 8 == 0, size % 4 == 0, offset <= buffer.descriptor.size, size <= buffer.descriptor.size - offset else {
            throw GPUError.validation("Invalid Metal buffer mapping")
        }
        buffers[id]?.mapping = .init(buffer: id, mode: mode, offset: offset, size: size, generation: 0)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func readMappedBytes(id: ResourceID, offset: UInt64, size: UInt64) throws -> [UInt8] {
        try requireActive()
        #if canImport(Metal)
        guard let buffer = buffers[id], let mapping = buffer.mapping else {
            throw GPUError.validation("Metal buffer has no active mapping")
        }
        let length = try mapping.accessSize(offset: offset, size: size)
        let source = buffer.native.contents().advanced(by: Int(offset))
        return Array(UnsafeRawBufferPointer(start: source, count: Int(length)))
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func writeMappedBytes(id: ResourceID, offset: UInt64, bytes: [UInt8]) throws {
        try requireActive()
        #if canImport(Metal)
        guard let buffer = buffers[id], let mapping = buffer.mapping, mapping.mode == .write else {
            throw GPUError.validation("Metal buffer has no writable mapping")
        }
        _ = try mapping.accessSize(offset: offset, size: UInt64(bytes.count))
        if !bytes.isEmpty {
            bytes.withUnsafeBytes { source in
                buffer.native.contents().advanced(by: Int(offset))
                    .copyMemory(from: source.baseAddress!, byteCount: source.count)
            }
        }
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func endBufferMapping(id: ResourceID) {
        #if canImport(Metal)
        buffers[id]?.mapping = nil
        #endif
    }

    public func allocateQuerySet(id: ResourceID, descriptor: QuerySetDescriptor) throws {
        try requireActive()
        #if canImport(Metal)
        try descriptor.validate(against: capabilities)
        guard querySets[id] == nil else {
            throw GPUError.validation("Invalid Metal query set allocation")
        }
        let storage: QuerySet.Storage
        switch descriptor.type {
        case .timestamp:
            guard let timestampCounters else { throw GPUError.unsupported("Metal timestamp counters are unavailable") }
            let sampleDescriptor = MTLCounterSampleBufferDescriptor()
            sampleDescriptor.label = descriptor.label
            sampleDescriptor.counterSet = timestampCounters
            sampleDescriptor.sampleCount = Int(descriptor.count)
            sampleDescriptor.storageMode = .shared
            storage = .timestamp(try native.makeCounterSampleBuffer(descriptor: sampleDescriptor))
        case .occlusion:
            guard let visibility = native.makeBuffer(length: Int(descriptor.count) * 8, options: .storageModePrivate)
            else { throw GPUError.outOfMemory }
            visibility.label = descriptor.label
            storage = .occlusion(visibility)
        }
        querySets[id] = QuerySet(descriptor: descriptor, storage: storage)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyQuerySet(id: ResourceID) {
        #if canImport(Metal)
        querySets.removeValue(forKey: id)
        #endif
    }

    public func close(owner: UInt64) {
        #if canImport(Metal)
        buffers = buffers.filter { $0.key.device != owner }
        querySets = querySets.filter { $0.key.device != owner }
        textureViews = textureViews.filter { $0.key.device != owner }
        textures = textures.filter { $0.key.device != owner }
        externalTextures = externalTextures.filter { $0.key.device != owner }
        samplers = samplers.filter { $0.key.device != owner }
        pipelines = pipelines.filter { $0.key.device != owner }
        computePipelines = computePipelines.filter { $0.key.device != owner }
        renderPipelines = renderPipelines.filter { $0.key.device != owner }
        #endif
    }

    public func allocatePlanarExternalTexture(id: ResourceID, descriptor: PlanarExternalTextureDescriptor) throws {
        try requireActive()
        #if canImport(Metal)
        guard externalTextures[id] == nil, descriptor.planes.count == descriptor.format.planeComponents.count else {
            throw GPUError.validation("Invalid Metal external texture descriptor")
        }
        var first: (width: UInt32, height: UInt32)?
        for (index, plane) in descriptor.planes.enumerated() {
            let view = textureViews[plane]?.resource
            guard plane.device == id.device, let texture = textures[view?.texture ?? plane] else {
                throw GPUError.validation("Unknown Metal external texture plane")
            }
            let extent = try descriptor.planeExtent(at: index, texture: texture.descriptor, view: view, capabilities: capabilities)
            if index == 0 { first = extent }
        }
        guard let first else { throw GPUError.validation("External texture requires a first plane") }
        let words = try descriptor.parameterWords(planeWidth: first.width, planeHeight: first.height)
        externalTextures[id] = .init(planes: descriptor.planes, parameters: words)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyExternalTexture(id: ResourceID) {
        #if canImport(Metal)
        externalTextures.removeValue(forKey: id)
        #endif
    }

    public func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) throws {
        try requireActive()
        #if canImport(Metal)
        guard case .text(.msl, let source) = shader.artifact else {
            throw GPUError.validation("Metal requires a compiled MSL compute shader")
        }
        try capabilities.require(shader.storageTextureFeatures)
        if shader.requiresShaderInt64 { try capabilities.require([.shaderInt64]) }
        try capabilities.requireShaderInt64Atomics(minMax: shader.requiresShaderInt64AtomicMinMax,
                                                  allOps: shader.requiresShaderInt64AtomicAllOps)
        if shader.requiresSubgroups { try capabilities.require([.subgroups]) }
        if shader.requiresSubgroupUniformity { try capabilities.require([.subgroupUniformity]) }
        try capabilities.requireSubgroupSize(shader.requiredSubgroupSize)
        if shader.requiresSubgroupID { try capabilities.require([.subgroupID]) }
        if shader.requiresLinearIndexing { try capabilities.require([.linearIndexing]) }
        let reflected = try ComputePipeline(id: pipeline.id, shader: shader,
                                            nonFilteringSamplers: pipeline.nonFilteringSamplers)
        guard pipeline == reflected else {
            throw GPUError.validation("Metal compute pipeline layout does not match compiled shader")
        }
        guard computePipelines[pipeline.id] == nil else { throw GPUError.validation("Duplicate Metal compute pipeline") }
        try capabilities.requireImmediateSize(pipeline.immediateByteSize)
        if let immediate = shader.immediateLayout,
           immediate.metalBufferIndex != ComputeShaderArtifact.metalImmediateBufferIndex {
            throw GPUError.validation("Invalid Metal compute immediate slot")
        }
        var bufferIndices: [ComputeBindingSlot: Int] = [:]
        var occupied = Set<UInt32>()
        func addBuffers(_ slots: [ShaderBindingIndex], range: Range<UInt32>) throws {
            for (slot, index) in slots {
                guard let index, range.contains(index), occupied.insert(index).inserted, bufferIndices[slot] == nil else {
                    throw GPUError.validation("Invalid Metal compute buffer binding layout")
                }
                bufferIndices[slot] = Int(index)
            }
        }
        for binding in shader.storageBindings {
            try addBuffers(binding.indexedSlots(), range: 0..<ComputeShaderArtifact.storageBindingCapacity)
        }
        let uniformBase = ComputeShaderArtifact.uniformBindingBase
        let uniformLimit = uniformBase + ComputeShaderArtifact.uniformBindingCapacity
        for binding in shader.uniformBindings {
            try addBuffers(binding.indexedSlots(), range: uniformBase..<uniformLimit)
        }

        func addResource(
            _ binding: IRResourceBinding, array: ComputeShaderArtifact.BindingArrayLayout?, index: UInt32?,
            range: Range<UInt32>, occupied: inout Set<UInt32>, slots: inout [ComputeBindingSlot: Int]
        ) throws {
            guard let index, range.contains(index), (array?.count ?? 1) <= range.upperBound - index else {
                throw GPUError.validation("Invalid Metal resource index range")
            }
            let elements = try binding.computeSlots(array: array, limit: range.upperBound - range.lowerBound)
            for (offset, element) in elements.enumerated() {
                let native = index + UInt32(offset)
                guard occupied.insert(native).inserted, slots[element] == nil, bufferIndices[element] == nil else {
                    throw GPUError.validation("Overlapping Metal resource bindings")
                }
                slots[element] = Int(native)
            }
        }

        var textureIndices: [ComputeBindingSlot: Int] = [:]
        var occupiedTextures = Set<UInt32>()
        for binding in shader.sampledTextures {
            try addResource(binding.binding, array: binding.bindingArray, index: binding.flatBindingIndex,
                            range: 0..<ComputeShaderArtifact.sampledTextureBindingCapacity,
                            occupied: &occupiedTextures, slots: &textureIndices)
        }

        let storageBase = ComputeShaderArtifact.sampledTextureBindingCapacity
        let storageLimit = storageBase + ComputeShaderArtifact.storageTextureBindingCapacity
        for binding in shader.storageTextures {
            try addResource(binding.binding, array: binding.bindingArray, index: binding.flatBindingIndex,
                            range: storageBase..<storageLimit, occupied: &occupiedTextures, slots: &textureIndices)
        }

        var samplerIndices: [ComputeBindingSlot: Int] = [:]
        var occupiedSamplers = Set<UInt32>()
        for binding in shader.samplerBindings {
            try addResource(binding.binding, array: binding.bindingArray, index: binding.flatBindingIndex,
                            range: 0..<ComputeShaderArtifact.samplerBindingCapacity,
                            occupied: &occupiedSamplers, slots: &samplerIndices)
        }
        guard samplerIndices.keys.allSatisfy({ textureIndices[$0] == nil }) else {
            throw GPUError.validation("Overlapping Metal texture and sampler bindings")
        }
        try validateExternalTextureLayouts(shader.externalTextureLayouts, bindings: shader.externalTextures,
                                           textureIndices: occupiedTextures, uniformIndices: occupied)

        let size = shader.workgroupSize
        let threads = MTLSize(width: Int(size.x), height: Int(size.y), depth: Int(size.z))
        let limit = native.maxThreadsPerThreadgroup
        guard (1...limit.width).contains(threads.width),
              (1...limit.height).contains(threads.height),
              (1...limit.depth).contains(threads.depth)
        else { throw GPUError.validation("Workgroup dimensions exceed Metal device limits") }
        try capabilities.requireWorkgroupStorageSize(shader.workgroupStorageByteSize)
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
        // Native compilation can add scratch space, including validation metadata.
        try capabilities.requireWorkgroupStorageSize(UInt64(state.staticThreadgroupMemoryLength))
        if let required = shader.requiredSubgroupSize, state.threadExecutionWidth != Int(required) {
            throw GPUError.validation("Requested subgroup size does not match the Metal pipeline SIMD width")
        }
        guard threads.width * threads.height * threads.depth <= state.maxTotalThreadsPerThreadgroup else {
            throw GPUError.validation("Workgroup size exceeds Metal pipeline limits")
        }
        let runtimeLengths = try runtimeLengthRequirements(shader.storageBindings)
        guard runtimeLengths.isEmpty == (shader.metalRuntimeLengthBuffer == nil),
              runtimeLengths.allSatisfy({ $0.1 > 0 }),
              shader.metalRuntimeLengthBuffer == nil ||
                  shader.metalRuntimeLengthBuffer == ComputeShaderArtifact.storageBindingCapacity
        else { throw GPUError.validation("Invalid Metal runtime array layout") }
        computePipelines[pipeline.id] = Pipeline(state: state, threadsPerThreadgroup: threads,
                                                bufferIndices: bufferIndices, textureIndices: textureIndices,
                                                samplerIndices: samplerIndices, runtimeLengths: runtimeLengths,
                                                lengthBufferIndex: shader.metalRuntimeLengthBuffer,
                                                immediateBufferIndex: shader.immediateLayout?.metalBufferIndex,
                                                externalTextures: shader.externalTextureLayouts)
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
        guard textures[id] == nil, buffers[id] == nil, samplers[id] == nil, textureViews[id] == nil else {
            throw GPUError.validation("Duplicate Metal resource")
        }
        let metal = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: descriptor.format.metal, width: Int(descriptor.width),
            height: Int(descriptor.height), mipmapped: descriptor.mipLevelCount > 1)
        let samples = Int(descriptor.sampleCount)
        guard native.supportsTextureSampleCount(samples) else {
            throw GPUError.unsupported("Metal does not support the texture sample count")
        }
        if descriptor.dimension == .d1 {
            metal.textureType = .type1D
        } else if descriptor.dimension == .d3 {
            metal.textureType = .type3D
            metal.depth = Int(descriptor.depth)
        } else if samples > 1 {
            metal.textureType = .type2DMultisample
        } else if descriptor.arrayLayerCount > 1 {
            metal.textureType = .type2DArray
            metal.arrayLength = Int(descriptor.arrayLayerCount)
        }
        metal.sampleCount = samples
        metal.mipmapLevelCount = Int(descriptor.mipLevelCount)
        metal.storageMode = .private
        metal.usage = clearsTextureWithBlit(descriptor) ? [] : .renderTarget
        if descriptor.usage.contains(.textureBinding) ||
            (supportsPresentation && descriptor.format.isNormalizedColor && descriptor.usage.contains(.copySource)) {
            metal.usage.insert(.shaderRead)
        }
        if descriptor.format.hasStencil || descriptor.format.compression != nil ||
            ((descriptor.format.isNormalizedColor || descriptor.format.isSRGB8Color) && descriptor.usage.contains(.copyDestination)) {
            metal.usage.insert(.pixelFormatView)
        }
        if descriptor.usage.contains(.storageBinding) { metal.usage.formUnion([.shaderRead, .shaderWrite]) }
        if descriptor.usage.contains(.storageAtomic) {
            if #available(macOS 14, iOS 17, *) {
                metal.usage.formUnion([.shaderRead, .shaderWrite, .shaderAtomic])
            }
        }
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

    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>) throws {
        try allocateTextureView(id: id, texture: texture, mipLevels: mipLevels, aspect: .all)
    }

    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>,
                                    aspect: TextureViewAspect) throws {
        try allocateTextureView(id: id, texture: texture, descriptor: .init(
            baseMipLevel: mipLevels.lowerBound, mipLevelCount: mipLevels.upperBound - mipLevels.lowerBound, aspect: aspect))
    }

    public func allocateTextureView(id: ResourceID, texture: ResourceID, descriptor: TextureViewDescriptor) throws {
        try requireActive()
        #if canImport(Metal)
        guard textureViews[id] == nil, buffers[id] == nil, textures[id] == nil, samplers[id] == nil,
              id.device == texture.device, let parent = textures[texture] else {
            throw GPUError.validation("Invalid Metal texture view")
        }
        let mipLevels = try descriptor.mipLevels(in: parent.descriptor)
        let layers = try descriptor.arrayLayers(in: parent.descriptor)
        let dimension = descriptor.resolvedDimension(in: parent.descriptor)
        let aspect = descriptor.aspect
        let view: any MTLTexture
        let multisampled = parent.descriptor.sampleCount > 1
        let format: MTLPixelFormat = aspect == .stencilOnly && parent.descriptor.format.hasDepth ? .x32_stencil8 : parent.native.pixelFormat
        if (multisampled || dimension == .d1) && format == parent.native.pixelFormat {
            // These textures have one mip and can share the handle unless reinterpreted.
            view = parent.native
        } else {
            let levels = Int(mipLevels.lowerBound)..<Int(mipLevels.upperBound)
            let slices = Int(layers.lowerBound)..<Int(layers.upperBound)
            let type: MTLTextureType = switch dimension {
            case .d2: multisampled ? .type2DMultisample : .type2D
            case .d2Array: .type2DArray
            case .cube: .typeCube
            case .cubeArray: .typeCubeArray
            case .d3: .type3D
            case .d1: .type1D
            }
            guard let nativeView = parent.native.makeTextureView(
                pixelFormat: format, textureType: type, levels: levels, slices: slices) else {
                throw GPUError.backendFailure("Metal could not create a texture view")
            }
            view = nativeView
        }
        textureViews[id] = .init(resource: .init(texture: texture, levels: mipLevels, aspect: aspect,
                                                layers: layers, dimension: dimension), native: view)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyTextureView(id: ResourceID) {
        #if canImport(Metal)
        textureViews.removeValue(forKey: id)
        #endif
    }

    public func allocateSampler(id: ResourceID, descriptor: SamplerDescriptor) throws {
        try requireActive()
        #if canImport(Metal)
        guard samplers[id] == nil, buffers[id] == nil, textures[id] == nil, textureViews[id] == nil else {
            throw GPUError.validation("Duplicate Metal resource")
        }
        func addressMode(_ mode: SamplerAddressMode) -> MTLSamplerAddressMode {
            switch mode {
            case .clampToEdge: .clampToEdge
            case .repeat: .repeat
            case .mirrorRepeat: .mirrorRepeat
            }
        }
        let metal = MTLSamplerDescriptor()
        metal.sAddressMode = addressMode(descriptor.addressModeU)
        metal.tAddressMode = addressMode(descriptor.addressModeV)
        metal.minFilter = descriptor.minFilter == .linear ? .linear : .nearest
        metal.magFilter = descriptor.magFilter == .linear ? .linear : .nearest
        metal.mipFilter = descriptor.mipmapFilter == .linear ? .linear : .nearest
        if let compare = descriptor.compare { metal.compareFunction = compare.metal }
        guard let sampler = native.makeSamplerState(descriptor: metal) else { throw GPUError.outOfMemory }
        samplers[id] = .init(native: sampler, descriptor: descriptor)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroySampler(id: ResourceID) {
        #if canImport(Metal)
        samplers.removeValue(forKey: id)
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
        if fragment.readsPrimitiveIndex { try capabilities.require([.primitiveIndex]) }
        if fragment.requiresBarycentrics { try capabilities.require([.shaderBarycentrics]) }
        guard vertex.immediateLayout.map({ $0.metalBufferIndex == RenderStageArtifact.metalVertexImmediateBufferIndex }) ?? true,
              fragment.immediateLayout.map({ $0.metalBufferIndex == RenderStageArtifact.metalFragmentImmediateBufferIndex }) ?? true else {
            throw GPUError.validation("Invalid Metal render immediate slot")
        }
        if fragment.earlyDepthTest != nil { try capabilities.require([.shaderEarlyDepthTest]) }
        if fragment.requiresSubgroups || vertex.requiresSubgroups { try capabilities.require([.subgroups]) }
        if vertex.requiresSubgroupUniformity || fragment.requiresSubgroupUniformity {
            try capabilities.require([.subgroupUniformity])
        }
        if vertex.requiresShaderInt64 || fragment.requiresShaderInt64 { try capabilities.require([.shaderInt64]) }
        try capabilities.requireShaderInt64Atomics(
            minMax: vertex.requiresShaderInt64AtomicMinMax || fragment.requiresShaderInt64AtomicMinMax,
            allOps: vertex.requiresShaderInt64AtomicAllOps || fragment.requiresShaderInt64AtomicAllOps)
        if fragment.usesDualSourceBlending || pipeline.colorTargets.values.contains(where: { $0.colorState.usesSecondBlendSource }) {
            try capabilities.require([.dualSourceBlending])
        }
        if pipeline.primitive.unclippedDepth { try capabilities.require([.depthClipControl]) }
        let features = vertex.storageTextures.requiredStorageTextureFeatures
            .union(fragment.storageTextures.requiredStorageTextureFeatures)
        try capabilities.require(features)
        let reflected = try RenderPipeline(id: pipeline.id, colorFormat: pipeline.colorFormat,
                                           vertexBuffers: pipeline.vertexBuffers, depthState: pipeline.depthState,
                                           sampleCount: pipeline.sampleCount, sampleMask: pipeline.sampleMask,
                                           alphaToCoverageEnabled: pipeline.alphaToCoverageEnabled, primitive: pipeline.primitive,
                                           colorState: pipeline.colorState, additionalColorTargets: pipeline.additionalColorTargets,
                                           vertex: vertex, fragment: fragment,
                                           nonFilteringSamplers: pipeline.nonFilteringSamplers, capabilities: capabilities)
        guard pipeline == reflected else {
            throw GPUError.validation("Metal render pipeline layout does not match compiled shaders")
        }
        let needsAtomicGuard = fragment.storageTextures.contains { $0.access.contains(.atomic) }
        let expectedGuard = needsAtomicGuard ? RenderStageArtifact.metalTextureAtomicGuardSlot : nil
        guard vertex.metalTextureAtomicGuardBufferIndex == nil,
              fragment.metalTextureAtomicGuardBufferIndex == expectedGuard else {
            throw GPUError.validation("Invalid Metal fragment texture atomic guard layout")
        }
        let uniformBase = ComputeShaderArtifact.uniformBindingBase
        let uniformLimit = uniformBase + ComputeShaderArtifact.uniformBindingCapacity
        for stage in [vertex, fragment] {
            let uniformSlots = try stage.uniformBindings.flatMap { try $0.indexedSlots() }
            let indices = uniformSlots.compactMap(\.index)
            guard indices.count == uniformSlots.count, Set(indices).count == indices.count,
                  indices.allSatisfy({ (uniformBase..<uniformLimit).contains($0) && Int($0) >= pipeline.vertexBuffers.count })
            else { throw GPUError.validation("Invalid Metal render uniform layout") }
            let storageBufferSlots = try stage.storageBindings.flatMap { try $0.indexedSlots() }
            let storageBufferIndices = storageBufferSlots.compactMap(\.index)
            let bufferBase = RenderStageArtifact.metalStorageBufferBase
            let bufferLimit = RenderStageArtifact.metalStorageLengthBufferIndex
            let lengths = try runtimeLengthRequirements(stage.storageBindings)
            let expectedLengthBuffer = lengths.isEmpty ? nil : bufferLimit
            guard storageBufferIndices.count == storageBufferSlots.count,
                  Set(storageBufferIndices).count == storageBufferIndices.count,
                  storageBufferIndices.allSatisfy({ (bufferBase..<bufferLimit).contains($0) }),
                  lengths.allSatisfy({ $0.1 > 0 }), stage.metalRuntimeLengthBuffer == expectedLengthBuffer else {
                throw GPUError.validation("Invalid Metal render storage buffer layout")
            }
            let textureSlots = try stage.sampledTextures.flatMap { try $0.indexedSlots() }
            let textureIndices = textureSlots.compactMap(\.index)
            guard textureIndices.count == textureSlots.count, Set(textureIndices).count == textureIndices.count,
                  textureIndices.allSatisfy({ $0 < ComputeShaderArtifact.sampledTextureBindingCapacity })
            else { throw GPUError.validation("Invalid Metal render sampled texture layout") }
            let storageBase = ComputeShaderArtifact.sampledTextureBindingCapacity
            let storageLimit = storageBase + ComputeShaderArtifact.storageTextureBindingCapacity
            let storageSlots = try stage.storageTextures.flatMap { try $0.indexedSlots() }
            let storageIndices = storageSlots.compactMap(\.index)
            guard storageIndices.count == storageSlots.count, Set(storageIndices).count == storageIndices.count,
                  storageIndices.allSatisfy({ (storageBase..<storageLimit).contains($0) })
            else { throw GPUError.validation("Invalid Metal render storage texture layout") }
            try validateExternalTextureLayouts(stage.externalTextureLayouts, bindings: stage.externalTextures,
                                               textureIndices: Set(textureIndices + storageIndices), uniformIndices: Set(indices))
            let samplerSlots = try stage.samplerBindings.flatMap { try $0.indexedSlots() }
            let samplerIndices = samplerSlots.compactMap(\.index)
            guard samplerIndices.count == samplerSlots.count, Set(samplerIndices).count == samplerIndices.count,
                  samplerIndices.allSatisfy({ $0 < ComputeShaderArtifact.samplerBindingCapacity })
            else { throw GPUError.validation("Invalid Metal render sampler layout") }
        }
        do {
            let vertexOptions: MTLCompileOptions?
            let isPointList = pipeline.primitive.topology == .pointList
            if isPointList || vertex.invariantPosition {
                let options = MTLCompileOptions()
                options.preserveInvariance = vertex.invariantPosition
                if isPointList {
                    options.preprocessorMacros = [RenderStageArtifact.metalPointTopologyMacro: NSNumber(value: 1)]
                }
                vertexOptions = options
            } else {
                vertexOptions = nil
            }
            let vs = try native.makeLibrary(source: vertexSource, options: vertexOptions)
            let activeSamples = (UInt32(1) << pipeline.sampleCount) - 1
            let sampleMask = pipeline.sampleMask & activeSamples
            let fragmentOptions: MTLCompileOptions?
            if sampleMask != activeSamples {
                let options = MTLCompileOptions()
                options.preprocessorMacros = [RenderStageArtifact.metalPipelineSampleMaskMacro: NSNumber(value: sampleMask)]
                fragmentOptions = options
            } else {
                fragmentOptions = nil
            }
            let fs = try native.makeLibrary(source: fragmentSource, options: fragmentOptions)
            guard let vertexFunction = vs.makeFunction(name: vertex.entryPoint),
                  let fragmentFunction = fs.makeFunction(name: fragment.entryPoint) else {
                throw GPUError.backendFailure("Compiled Metal render entry point missing")
            }
            let descriptor = MTLRenderPipelineDescriptor()
            let samples = Int(pipeline.sampleCount)
            guard native.supportsTextureSampleCount(samples) else {
                throw GPUError.unsupported("Metal does not support the render pipeline sample count")
            }
            descriptor.rasterSampleCount = samples
            descriptor.isAlphaToCoverageEnabled = pipeline.alphaToCoverageEnabled
            descriptor.inputPrimitiveTopology = pipeline.primitive.topology.metal.topologyClass
            descriptor.vertexFunction = vertexFunction
            descriptor.fragmentFunction = fragmentFunction
            for (location, target) in pipeline.colorTargets {
                let color = descriptor.colorAttachments[Int(location)]!
                color.pixelFormat = target.format.metal
                color.writeMask = pipeline.colorWriteMask(at: location).metal
                if let blend = target.colorState.blend {
                    color.isBlendingEnabled = true
                    color.rgbBlendOperation = blend.color.operation.metal
                    color.sourceRGBBlendFactor = blend.color.sourceFactor.metal
                    color.destinationRGBBlendFactor = blend.color.destinationFactor.metal
                    color.alphaBlendOperation = blend.alpha.operation.metal
                    color.sourceAlphaBlendFactor = blend.alpha.sourceFactor.metal
                    color.destinationAlphaBlendFactor = blend.alpha.destinationFactor.metal
                }
            }
            let depth: (any MTLDepthStencilState)?
            if let state = pipeline.depthState {
                if state.format.hasDepth { descriptor.depthAttachmentPixelFormat = state.format.metal }
                let depthDescriptor = MTLDepthStencilDescriptor()
                depthDescriptor.depthCompareFunction = state.compare.metal
                depthDescriptor.isDepthWriteEnabled = state.writeEnabled
                if state.format.hasStencil {
                    descriptor.stencilAttachmentPixelFormat = state.format.metal
                    let stencil = state.stencil ?? .init()
                    let writeMask = stencil.writes(cullMode: pipeline.primitive.cullMode) ? stencil.writeMask : 0
                    depthDescriptor.frontFaceStencil = stencil.front.metal(readMask: stencil.readMask, writeMask: writeMask)
                    depthDescriptor.backFaceStencil = stencil.back.metal(readMask: stencil.readMask, writeMask: writeMask)
                }
                guard let depthState = native.makeDepthStencilState(descriptor: depthDescriptor) else {
                    throw GPUError.backendFailure("Metal could not create a depth state")
                }
                depth = depthState
            } else {
                depth = nil
            }
            if !pipeline.vertexBuffers.isEmpty {
                let vertexDescriptor = MTLVertexDescriptor()
                for (slot, layout) in pipeline.vertexBuffers.enumerated() {
                    let buffer = vertexDescriptor.layouts[slot]!
                    buffer.stride = layout.stride == 0 ? Int(layout.attributeEnd) : Int(layout.stride)
                    if layout.stride == 0 {
                        buffer.stepFunction = .constant
                    } else {
                        buffer.stepFunction = layout.stepMode == .instance ? .perInstance : .perVertex
                    }
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
            let state = try native.makeRenderPipelineState(descriptor: descriptor)
            renderPipelines[pipeline.id] = RenderState(pipeline: state, depth: depth, writesDepth: fragment.writesDepth,
                                                       vertexRuntimeLengths: try runtimeLengthRequirements(vertex.storageBindings),
                                                       fragmentRuntimeLengths: try runtimeLengthRequirements(fragment.storageBindings),
                                                       vertexExternalTextures: vertex.externalTextureLayouts,
                                                       fragmentExternalTextures: fragment.externalTextureLayouts,
                                                       fragmentTextureAtomicGuardBufferIndex: fragment.metalTextureAtomicGuardBufferIndex,
                                                       vertexImmediateBufferIndex: vertex.immediateLayout?.metalBufferIndex,
                                                       fragmentImmediateBufferIndex: fragment.immediateLayout?.metalBufferIndex)
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

    public func present(_ submission: Submission, source: ResourceID) async throws {
        try requireActive()
        #if canImport(Metal) && canImport(QuartzCore)
        guard let surface else { throw GPUError.unsupported("Metal backend does not have a presentation surface") }
        guard !presenting else { throw GPUError.validation("Metal surface already has a pending frame") }
        guard submission.resources.contains(source), let texture = textures[source],
              texture.descriptor.format.isNormalizedColor, texture.descriptor.sampleCount == 1,
              texture.descriptor.dimension == .d2, texture.descriptor.arrayLayerCount == 1,
              texture.descriptor.usage.contains(.copySource) else {
            throw GPUError.validation("Unpinned or invalid Metal presentation source")
        }
        let descriptor = texture.descriptor
        presenting = true
        defer { presenting = false }
        let pipeline: any MTLRenderPipelineState
        if let surfacePipeline {
            pipeline = surfacePipeline
        } else {
            pipeline = try MetalSurfaceCopy.makePipeline(device: native)
            surfacePipeline = pipeline
        }
        let drawable = try surface.acquire(width: descriptor.width, height: descriptor.height)
        guard drawable.texture.width == Int(descriptor.width), drawable.texture.height == Int(descriptor.height) else {
            throw GPUError.backendFailure("Metal drawable size did not match the presentation source")
        }
        guard let commands = queue.makeCommandBuffer() else { throw GPUError.outOfMemory }
        let clear = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        var initialized: [ResourceID: Set<TextureSubresource>] = [:]
        try initializeTexture(source, level: 0, on: commands, initialized: &initialized)
        let encoder = try renderEncoder(commands, target: drawable.texture, clear: clear)
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(texture.native, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commands.present(drawable)
        try await commit(Batch(commandBuffer: commands, buffers: [], pipelines: [], textures: [texture.native],
                               samplers: [], renderPipelines: [.init(pipeline: pipeline, depth: nil, writesDepth: false,
                                                       vertexRuntimeLengths: [], fragmentRuntimeLengths: [],
                                                       vertexExternalTextures: [], fragmentExternalTextures: [],
                                                       fragmentTextureAtomicGuardBufferIndex: nil,
                                                       vertexImmediateBufferIndex: nil, fragmentImmediateBufferIndex: nil)],
                               drawable: drawable), initialized: initialized)
        #else
        throw GPUError.unsupported("Metal presentation is unavailable")
        #endif
    }

    public func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        try requireActive()
        #if canImport(Metal)
        let owner = submission.resources.first?.device ??
            commands.lazy.compactMap { $0.recordedPass.command.computeDispatch?.pipeline.id.device }.first ?? 0
        let descriptors = buffers.mapValues(\.descriptor)
        let queryDescriptors = querySets.mapValues(\.descriptor)
        let textureDescriptors = textures.mapValues(\.descriptor)
        let samplerDescriptors = samplers.mapValues(\.descriptor)
        let views = textureViews.mapValues(\.resource)
        let externalPlanes = externalTextures.mapValues(\.planes)
        let pinned = Set(submission.resources)
        for id in submission.resources {
            guard buffers[id] != nil || textures[id] != nil || views[id] != nil || samplers[id] != nil ||
                  querySets[id] != nil || externalPlanes[id] != nil else {
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
                                  maximumViewportDimension: capabilities.maxTextureDimension2D, querySets: queryDescriptors,
                                  features: capabilities.features, externalTextures: Set(externalPlanes.keys),
                                  externalPlanes: externalPlanes)
            let command = recorded.recordedPass.command
            if case .dispatch(let id, _, _, _) = command, pipelines[id] == nil {
                throw GPUError.validation("Unknown Metal diagnostic pipeline")
            }
            if let dispatch = command.computeDispatch, computePipelines[dispatch.pipeline.id] == nil {
                throw GPUError.validation("Unknown Metal compute pipeline")
            }
            switch command {
            case .render(let pipeline, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
                 .renderIndexed(let pipeline, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
                 .renderIndirect(let pipeline, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _):
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
        for range in queryCommandRanges(commands) {
            let (batch, reads) = try encodeCommands(commands[range], resources: submission.resources,
                                                    initialized: &initialized, writtenQueries: &writtenQueries)
            encoded.append((batch, reads, initialized, writtenQueries))
        }
        var snapshots: [[UInt32]] = []
        for item in encoded {
            try await commit(item.batch, initialized: item.initialized, writtenQueries: item.written)
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

    private func encodeCommands(
        _ commands: ArraySlice<BufferCommand>, resources: [ResourceID],
        initialized: inout [ResourceID: Set<TextureSubresource>], writtenQueries: inout [ResourceID: Set<UInt32>]
    ) throws -> (Batch, [(any MTLBuffer, Int)]) {
        guard let commandBuffer = queue.makeCommandBuffer() else { throw GPUError.outOfMemory }
        var retained = resources.compactMap { buffers[$0]?.native ?? querySets[$0]?.visibility }
        var retainedCopyViews: [any MTLTexture] = []
        var retainedPipelines: [any MTLComputePipelineState] = []
        var retainedRenderPipelines: [RenderState] = []
        var reads: [(any MTLBuffer, Int)] = []
        func encodeDraw(
            _ pipeline: RenderPipeline, target: ResourceID, colorLoad: RenderColorLoadAction, level: UInt32, layer: UInt32?,
            depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?, additionalColors: [UInt32: RenderColorAttachment],
            vertexBuffers: [VertexBufferBinding], uniforms: [ComputeBindingSlot: ComputeBufferBinding],
            storageBuffers: [ComputeBindingSlot: ComputeBufferBinding], textures: [ComputeBindingSlot: ResourceID],
            samplers: [ComputeBindingSlot: ResourceID], blendConstant: RenderClearColor, raster: RenderRasterState,
            immediates: [UInt32], drawEnabled: Bool, timestamps: PassTimestampWrites?, occlusion: PassOcclusionQuery?,
            draw: (any MTLRenderCommandEncoder) -> Void
        ) throws {
            let state = renderPipelines[pipeline.id]!
            if drawEnabled { try initializeBoundTextures(textures, on: commandBuffer, initialized: &initialized) }
            try initializeLoadedAttachments(target: target, level: level, depthSlice: layer, colorLoad: colorLoad, depth: depth,
                                            additionalColors: additionalColors, on: commandBuffer, initialized: &initialized)
            let vertexLengths = drawEnabled ? try runtimeLengthValues(state.vertexRuntimeLengths, bindings: storageBuffers) : []
            let fragmentLengths = drawEnabled ? try runtimeLengthValues(state.fragmentRuntimeLengths, bindings: storageBuffers) : []
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
                retained.append(try encodeTimestampMarker(commandBuffer, writes: .init(
                    querySet: timestamps.querySet, beginningOfPassWriteIndex: start)))
            }
            let encoder = try renderEncoder(commandBuffer, target: nativeTexture(target), level: level, layer: layer ?? 0,
                                            clear: clear, depth: depth, resolve: resolve, additionalColors: additionalColors,
                                            timestamps: drawEnabled ? timestamps : nil, visibility: visibility)
            if drawEnabled {
                bindRender(encoder, state: state, pipeline: pipeline, vertexBuffers: vertexBuffers, uniforms: uniforms,
                           storageBuffers: storageBuffers, vertexLengths: vertexLengths, fragmentLengths: fragmentLengths,
                           boundTextures: textures, samplerBindings: samplers, blendConstant: blendConstant, raster: raster)
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
            if !drawEnabled, let timestamps, let end = timestamps.endOfPassWriteIndex {
                retained.append(try encodeTimestampMarker(commandBuffer, writes: .init(
                    querySet: timestamps.querySet, endOfPassWriteIndex: end)))
            }
            retainedRenderPipelines.append(state)
            markRenderedAttachments(target: target, level: level, depthSlice: layer, depth: depth, resolve: resolve,
                                    additionalColors: additionalColors, initialized: &initialized)
        }

        for recorded in commands {
            let (command, timestamps, occlusion) = recorded.recordedPass
            if let timestamps { writtenQueries[timestamps.querySet, default: []].formUnion(timestamps.indices) }
            if let occlusion { writtenQueries[occlusion.querySet, default: []].insert(occlusion.queryIndex) }
            switch command {
            case .timestamped, .occlusionQueried:
                throw GPUError.validation("Invalid query pass nesting")
            case .resolveQuerySet(let id, let first, let count, let destination, let offset):
                if count == 0 { continue }
                let query = querySets[id]!
                try resolveQueries(query, written: query.written.union(writtenQueries[id, default: []]),
                                   range: first..<(first + count), destination: buffers[destination]!.native,
                                   offset: Int(offset), on: commandBuffer)
            case .write(let id, let offset, let words):
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
                let (pipeline, bindings, source, boundTextures, samplerBindings, immediates) = command.computeDispatch!
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
                    if dispatchValidator == nil { dispatchValidator = try MetalDispatchValidator(device: native) }
                    let validated = try dispatchValidator!.encode(
                        on: commandBuffer, device: native, source: buffers[buffer]!.native, offset: Int(offset),
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
            case .render(let pipeline, let target, let colorLoad, let vertexCount, let vertexBuffers, let firstVertex, let instances, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates):
                let drawEnabled = vertexCount > 0 && instances.count > 0 && !raster.hasEmptyScissor
                try encodeDraw(pipeline, target: target, colorLoad: colorLoad, level: targetMipLevel, layer: targetDepthSlice,
                               depth: depth, resolve: resolve, additionalColors: additionalColors, vertexBuffers: vertexBuffers,
                               uniforms: uniforms, storageBuffers: storageBuffers, textures: boundTextures, samplers: samplerBindings,
                               blendConstant: blendConstant, raster: raster, immediates: immediates, drawEnabled: drawEnabled,
                               timestamps: timestamps, occlusion: occlusion) { encoder in
                    encoder.drawPrimitives(type: pipeline.primitive.topology.metal.type, vertexStart: Int(firstVertex),
                                           vertexCount: Int(vertexCount), instanceCount: Int(instances.count),
                                           baseInstance: Int(instances.first))
                }
            case .renderIndexed(let pipeline, let target, let colorLoad, let index, let draw, let vertexBuffers, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates):
                let indexBuffer = buffers[index.buffer]!.native
                let byteOffset = Int(index.offset + UInt64(draw.firstIndex) * UInt64(index.format.byteSize))
                var indirect: (any MTLBuffer)?
                var drawEnabled = draw.indexCount > 0 && draw.instances.count > 0 && !raster.hasEmptyScissor
                if drawEnabled && pipeline.vertexBuffers.contains(where: { $0.stepMode == .vertex }) {
                    let capacity = fetchCapacity(pipeline.vertexBuffers, bindings: vertexBuffers, stepMode: .vertex)
                    if capacity > 0 {
                        if indexValidator == nil { indexValidator = try MetalIndexValidator(device: native) }
                        let args = try indexValidator!.encode(
                            on: commandBuffer, device: native, indices: indexBuffer, format: index.format,
                            byteOffset: byteOffset, draw: draw, maximumVertex: UInt32(capacity - 1),
                            primitiveRestart: pipeline.primitive.topology.isStrip)
                        indirect = args
                        retained.append(args)
                    } else {
                        drawEnabled = false
                    }
                }
                try encodeDraw(pipeline, target: target, colorLoad: colorLoad, level: targetMipLevel, layer: targetDepthSlice,
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
            case .renderIndirect(let pipeline, let target, let colorLoad, let arguments, let offset, let index, let vertexBuffers, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates):
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
                        indices: indices, scanIndices: pipeline.vertexBuffers.contains { $0.stepMode == .vertex },
                        primitiveRestart: pipeline.primitive.topology.isStrip)
                    retained.append(args)
                    validated = args
                }
                try encodeDraw(pipeline, target: target, colorLoad: colorLoad, level: targetMipLevel, layer: targetDepthSlice,
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
                                      let layer, let z, let depth):
                let texture = textures[source]!
                let extent = texture.descriptor.mipExtent(mipLevel)!
                let physical = texture.descriptor.mipPhysicalExtent(mipLevel)!
                let region = TextureCopyRegion(width: physical.width, height: physical.height, mipLevel: mipLevel,
                                               arrayLayer: layer, z: z, depth: depth)
                for plane in region.planes(in: texture.descriptor) {
                    try initializeTexture(source, level: mipLevel, layer: plane, on: commandBuffer, initialized: &initialized)
                }
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                let blockHeight = texture.descriptor.format.copyBlock(aspect: aspect)!.height
                let imageStride = Int(bytesPerRow) * Int(physical.height / blockHeight)
                let options: MTLBlitOption = switch aspect {
                case .stencilOnly where texture.descriptor.format.hasDepth: .stencilFromDepthStencil
                case .depthOnly where texture.descriptor.format.hasStencil: .depthFromDepthStencil
                default: []
                }
                encoder.copy(from: texture.native, sourceSlice: Int(layer), sourceLevel: Int(mipLevel),
                             sourceOrigin: MTLOrigin(x: 0, y: 0, z: Int(z)),
                             sourceSize: MTLSize(width: Int(extent.width), height: Int(extent.height), depth: Int(depth)),
                             to: buffers[destination]!.native, destinationOffset: Int(offset),
                             destinationBytesPerRow: Int(bytesPerRow),
                             destinationBytesPerImage: texture.descriptor.dimension == .d3 ? imageStride : 0,
                             options: options)
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
                encoder.copy(from: sourceTexture.native, sourceSlice: Int(region.arrayLayer), sourceLevel: Int(region.mipLevel),
                             sourceOrigin: MTLOrigin(x: Int(region.x), y: Int(region.y), z: Int(region.z)),
                             sourceSize: MTLSize(width: Int(region.width), height: Int(region.height), depth: Int(region.depth)),
                             to: destinationTexture, destinationSlice: Int(origin.arrayLayer), destinationLevel: Int(origin.mipLevel),
                             destinationOrigin: MTLOrigin(x: Int(origin.x), y: Int(origin.y), z: Int(origin.z)))
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
        let batch = Batch(
            commandBuffer: commandBuffer, buffers: retained, pipelines: retainedPipelines,
            textures: resources.compactMap { textures[$0]?.native ?? textureViews[$0]?.native } + retainedCopyViews,
            samplers: resources.compactMap { samplers[$0]?.native }, renderPipelines: retainedRenderPipelines, drawable: nil,
            counterSampleBuffers: resources.compactMap { querySets[$0]?.samples })
        return (batch, reads)
    }

    private func encodeUpload(
        _ bytes: UnsafeRawBufferPointer, to destination: any MTLBuffer, offset: Int, on commands: any MTLCommandBuffer
    ) throws -> any MTLBuffer {
        guard let staging = native.makeBuffer(length: bytes.count, options: .storageModeShared),
              let encoder = commands.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
        staging.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        encoder.copy(from: staging, sourceOffset: 0, to: destination, destinationOffset: offset, size: bytes.count)
        encoder.endEncoding()
        return staging
    }

    private func nativeTexture(_ id: ResourceID) -> any MTLTexture {
        textureViews[id]?.native ?? textures[id]!.native
    }

    private func markRenderedTexture(_ id: ResourceID, level: UInt32, depthSlice: UInt32? = nil,
                                     initialized: inout [ResourceID: Set<TextureSubresource>]) {
        let view = textureViews[id]?.resource
        initialized[view?.texture ?? id, default: []].insert(
            .init(level: view?.levels.lowerBound ?? level, layer: depthSlice ?? view?.layers.lowerBound ?? 0))
    }

    private func markRenderedAttachments(
        target: ResourceID, level: UInt32, depthSlice: UInt32?, depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?,
        additionalColors: [UInt32: RenderColorAttachment], initialized: inout [ResourceID: Set<TextureSubresource>]
    ) {
        markRenderedTexture(target, level: level, depthSlice: depthSlice, initialized: &initialized)
        if let resolve { markRenderedTexture(resolve.texture, level: resolve.mipLevel, initialized: &initialized) }
        for color in additionalColors.values {
            markRenderedTexture(color.texture, level: color.mipLevel, depthSlice: color.depthSlice, initialized: &initialized)
            if let resolve = color.resolve {
                markRenderedTexture(resolve.texture, level: resolve.mipLevel, initialized: &initialized)
            }
        }
        if let depth { markRenderedTexture(depth.texture, level: depth.mipLevel, initialized: &initialized) }
    }

    private func initializeLoadedAttachments(
        target: ResourceID, level: UInt32, depthSlice: UInt32?, colorLoad: RenderColorLoadAction,
        depth: RenderDepthAttachment?, additionalColors: [UInt32: RenderColorAttachment], on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        func initialize(_ id: ResourceID, level: UInt32, depthSlice: UInt32? = nil) throws {
            let view = textureViews[id]?.resource
            try initializeTexture(view?.texture ?? id, level: view?.levels.lowerBound ?? level,
                                  layer: depthSlice ?? view?.layers.lowerBound ?? 0, on: commands, initialized: &initialized)
        }
        if case .load = colorLoad { try initialize(target, level: level, depthSlice: depthSlice) }
        for attachment in additionalColors.values where attachment.loadAction == .load {
            try initialize(attachment.texture, level: attachment.mipLevel, depthSlice: attachment.depthSlice)
        }
        if let depth {
            let loadsDepth = depth.loadAction == .load || depth.loadAction == .readOnly
            let loadsStencil = depth.stencil?.loadAction == .load || depth.stencil?.loadAction == .readOnly
            if loadsDepth || loadsStencil { try initialize(depth.texture, level: depth.mipLevel) }
        }
    }

    private func initializeBoundTextures(
        _ bindings: [ComputeBindingSlot: ResourceID], on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        for id in Set(bindings.values.flatMap { externalTextures[$0]?.planes ?? [$0] }) {
            let view = textureViews[id]?.resource
            let parent = view?.texture ?? id
            let levels = view?.levels ?? 0..<textures[parent]!.descriptor.mipLevelCount
            let descriptor = textures[parent]!.descriptor
            for level in levels {
                let layers = descriptor.dimension == .d3 ? 0..<descriptor.mipDepth(level) :
                    (view?.layers ?? 0..<descriptor.arrayLayerCount)
                for layer in layers {
                    try initializeTexture(parent, level: level, layer: layer, on: commands, initialized: &initialized)
                }
            }
        }
    }

    private func validateExternalTextureLayouts(
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

    private func bindExternalTextures(
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

    private func stageTextureUpload(
        _ data: [UInt8], layout: TextureDataLayout, region: TextureCopyRegion, block: TextureCopyBlock
    ) throws -> (any MTLBuffer, TextureBufferLayout) {
        let footprint = try layout.footprint(region: region, block: block, dataSize: UInt64(data.count))
        let rowBytes = Int(footprint.rowBytes)
        let stride = (rowBytes + 3) & ~3
        let blockRows = Int(region.height / block.height)
        let (imageStride, imageOverflow) = stride.multipliedReportingOverflow(by: blockRows)
        let (size, sizeOverflow) = imageStride.multipliedReportingOverflow(by: Int(region.depth))
        guard !imageOverflow, !sizeOverflow, let bytesPerRow = UInt32(exactly: stride),
              let staging = native.makeBuffer(length: size, options: .storageModeShared) else {
            throw GPUError.outOfMemory
        }
        data.withUnsafeBytes { source in
            let input = source.baseAddress!.advanced(by: Int(layout.offset))
            let output = staging.contents()
            for image in 0..<Int(region.depth) {
                let imageStart = input.advanced(by: image * Int(footprint.imageStride))
                for row in 0..<blockRows {
                    let from = imageStart.advanced(by: row * Int(footprint.rowStride))
                    let to = output.advanced(by: image * imageStride + row * stride)
                    to.copyMemory(from: from, byteCount: rowBytes)
                }
            }
        }
        return (staging, .init(bytesPerRow: bytesPerRow))
    }

    private func encodeTextureUpload(
        _ source: any MTLBuffer, layout: TextureBufferLayout, destination: ResourceID, region: TextureCopyRegion,
        aspect: TextureViewAspect, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        let texture = textures[destination]!
        try prepareTextureWrite(destination, region: region, aspect: aspect, on: commands, initialized: &initialized)
        guard let encoder = commands.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
        let block = texture.descriptor.format.copyBlock(aspect: aspect)!
        let stride = Int(layout.rowStride(width: region.width, block: block))
        let imageStride = stride * Int(region.height / block.height)
        let volume = texture.descriptor.dimension == .d3
        // Metal buffer blits take the logical mip extent even when they transfer a complete compression block.
        let extent = texture.descriptor.mipExtent(region.mipLevel)!
        let size = MTLSize(width: Int(min(region.width, extent.width - region.x)),
                           height: Int(min(region.height, extent.height - region.y)),
                           depth: volume ? Int(region.depth) : 1)
        let origin = MTLOrigin(x: Int(region.x), y: Int(region.y), z: Int(region.z))
        for layer in 0..<(volume ? 1 : Int(region.depth)) {
            encoder.copy(from: source, sourceOffset: Int(layout.offset) + layer * imageStride,
                         sourceBytesPerRow: stride, sourceBytesPerImage: volume ? imageStride : 0,
                         sourceSize: size, to: texture.native, destinationSlice: Int(region.arrayLayer) + layer,
                         destinationLevel: Int(region.mipLevel), destinationOrigin: origin,
                         options: aspect == .stencilOnly && texture.descriptor.format.hasDepth ? .stencilFromDepthStencil : [])
        }
        encoder.endEncoding()
    }

    private func prepareTextureWrite(
        _ id: ResourceID, region: TextureCopyRegion, aspect: TextureViewAspect, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        let descriptor = textures[id]!.descriptor
        let fullPlane = (aspect != .stencilOnly || !descriptor.format.hasDepth) && region.covers(descriptor)
        for plane in region.planes(in: descriptor) {
            if fullPlane {
                initialized[id, default: []].insert(.init(level: region.mipLevel, layer: plane))
            } else {
                try initializeTexture(id, level: region.mipLevel, layer: plane, on: commands, initialized: &initialized)
            }
        }
    }

    private func clearsTextureWithBlit(_ descriptor: TextureDescriptor) -> Bool {
        if descriptor.dimension == .d1 || descriptor.format.compression != nil { return true }
        if descriptor.format.isSnorm8Color && !capabilities.features.contains(.snorm8Renderable) { return true }
        guard let feature = descriptor.format.hdrRenderFeature else { return false }
        return !capabilities.features.contains(feature)
    }

    private func initializeTexture(
        _ id: ResourceID, level: UInt32, layer: UInt32 = 0, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        let texture = textures[id]!
        // Only clear the subresource that could still expose uninitialized Metal memory.
        let subresource = TextureSubresource(level: level, layer: layer)
        guard !texture.initializedSubresources.contains(subresource),
              initialized[id, default: []].insert(subresource).inserted else { return }
        if clearsTextureWithBlit(texture.descriptor) {
            let logical = texture.descriptor.mipExtent(level)!
            let physical = texture.descriptor.mipPhysicalExtent(level)!
            let block = texture.descriptor.format.copyBlock(aspect: .all)!
            let rowBytes = Int(block.rowBytes(width: physical.width))
            let size = rowBytes * Int(physical.height / block.height)
            let fillSize = (size + 3) & ~3
            guard let zeroes = native.makeBuffer(length: fillSize, options: .storageModePrivate),
                  let encoder = commands.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
            encoder.fill(buffer: zeroes, range: 0..<fillSize, value: 0)
            let volume = texture.descriptor.dimension == .d3
            let origin = MTLOrigin(x: 0, y: 0, z: volume ? Int(layer) : 0)
            encoder.copy(from: zeroes, sourceOffset: 0, sourceBytesPerRow: rowBytes, sourceBytesPerImage: volume ? size : 0,
                         sourceSize: MTLSize(width: Int(logical.width), height: Int(logical.height), depth: 1),
                         to: texture.native, destinationSlice: volume ? 0 : Int(layer), destinationLevel: Int(level),
                         destinationOrigin: origin)
            encoder.endEncoding()
        } else if texture.descriptor.format.isDepthStencil {
            let pass = MTLRenderPassDescriptor()
            setDepthAttachment(pass, texture: texture.native, level: level, layer: layer,
                               loadAction: texture.descriptor.format.hasDepth ? .clear(0) : .unused,
                               stencilLoadAction: texture.descriptor.format.hasStencil ? .clear(0) : nil)
            guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { throw GPUError.outOfMemory }
            encoder.endEncoding()
        } else {
            let clear = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            try renderEncoder(commands, target: texture.native, level: level, layer: layer, clear: clear).endEncoding()
        }
    }

    private func commit(_ batch: Batch, initialized: [ResourceID: Set<TextureSubresource>],
                        writtenQueries: [ResourceID: Set<UInt32>] = [:]) async throws {
        guard nextBatch < UInt64.max else { throw GPUError.outOfMemory }
        nextBatch += 1
        let serial = nextBatch
        let completion = BackendCompletion()
        inFlight[serial] = batch
        let retirementHook = retirementHook
        batch.commandBuffer.addCompletedHandler { [self] buffer in
            let result: Result<Void, GPUError> = buffer.status == .completed ? .success(()) :
                .failure(.deviceLost("Metal command failed: \(String(describing: buffer.error))"))
            Task {
                if let retirementHook { await retirementHook() }
                await retire(serial, completion: completion, result: result)
            }
        }
        batch.commandBuffer.commit()
        for (id, levels) in initialized { textures[id]?.initializedSubresources.formUnion(levels) }
        for (id, indices) in writtenQueries { querySets[id]?.written.formUnion(indices) }
        do { try await completion.wait(timeout: timeout) }
        catch { terminal = true; throw error }
    }

    private func runtimeLengthRequirements(
        _ bindings: [ComputeShaderArtifact.StorageBinding]
    ) throws -> [(ComputeBindingSlot, UInt32, UInt32)] {
        try bindings.flatMap { binding -> [(ComputeBindingSlot, UInt32, UInt32)] in
            guard let stride = binding.runtimeArrayStride else { return [] }
            return try binding.binding.computeSlots(array: binding.bindingArray, limit: ComputeShaderArtifact.storageBindingCapacity)
                .map { ($0, stride, binding.runtimeArrayOffset) }
        }
    }

    private func runtimeLengthValues(
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

    private func bindRender(
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
        for (slot, binding) in vertexBuffers.enumerated() {
            encoder.setVertexBuffer(buffers[binding.buffer]!.native, offset: Int(binding.offset), index: slot)
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

    private func fetchCapacity(
        _ layouts: [VertexBufferLayout], bindings: [VertexBufferBinding], stepMode: VertexStepMode
    ) -> UInt64 {
        var count = MetalIndirectDrawValidator.maximumCapacity
        for (layout, binding) in zip(layouts, bindings) where layout.stepMode == stepMode {
            let available = buffers[binding.buffer]!.descriptor.size - binding.offset
            guard available >= layout.attributeEnd else { return 0 }
            if layout.stride > 0 {
                count = min(count, (available - layout.attributeEnd) / UInt64(layout.stride) + 1)
            }
        }
        return count
    }

    private func computeEncoder(
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

    /// An empty Metal encoder can be skipped, including its counter attachments.
    private func encodeTimestampMarker(
        _ commands: any MTLCommandBuffer, writes: PassTimestampWrites
    ) throws -> any MTLBuffer {
        let encoder = try computeEncoder(commands, timestamps: writes)
        defer { encoder.endEncoding() }
        return try encodeTimestampWork(encoder)
    }

    private func encodeTimestampWork(_ encoder: any MTLComputeCommandEncoder) throws -> any MTLBuffer {
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

    private func resolveQueries(
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

    private func renderEncoder(
        _ commands: any MTLCommandBuffer, target: any MTLTexture, level: UInt32 = 0, layer: UInt32 = 0, clear: MTLClearColor?,
        depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
        additionalColors: [UInt32: RenderColorAttachment] = [:], timestamps: PassTimestampWrites? = nil,
        visibility: (any MTLBuffer)? = nil
    ) throws -> any MTLRenderCommandEncoder {
        let pass = MTLRenderPassDescriptor()
        pass.visibilityResultBuffer = visibility
        if let timestamps {
            guard let attachment = pass.sampleBufferAttachments[0] else { throw GPUError.outOfMemory }
            attachment.sampleBuffer = querySets[timestamps.querySet]!.samples
            attachment.startOfVertexSampleIndex = timestamps.beginningOfPassWriteIndex.map(Int.init) ?? MTLCounterDontSample
            attachment.endOfVertexSampleIndex = MTLCounterDontSample
            attachment.startOfFragmentSampleIndex = MTLCounterDontSample
            attachment.endOfFragmentSampleIndex = timestamps.endOfPassWriteIndex.map(Int.init) ?? MTLCounterDontSample
        }
        func setColor(_ location: Int, texture: any MTLTexture, level: UInt32, layer: UInt32 = 0, clear: MTLClearColor?,
                      resolve: RenderResolveAttachment?) {
            let attachment = pass.colorAttachments[location]!
            attachment.texture = texture
            attachment.level = Int(level)
            if texture.textureType == .type3D {
                attachment.depthPlane = Int(layer)
            } else {
                attachment.slice = Int(layer)
            }
            attachment.loadAction = clear == nil ? .load : .clear
            attachment.storeAction = .store
            if let clear { attachment.clearColor = clear }
            if let resolve {
                attachment.resolveTexture = nativeTexture(resolve.texture)
                attachment.resolveLevel = Int(resolve.mipLevel)
                attachment.storeAction = .storeAndMultisampleResolve
            }
        }
        setColor(0, texture: target, level: level, layer: layer, clear: clear, resolve: resolve)
        for (location, attachment) in additionalColors {
            let clear = attachment.loadAction.clearColor.map {
                MTLClearColor(red: $0.red, green: $0.green, blue: $0.blue, alpha: $0.alpha)
            }
            setColor(Int(location), texture: nativeTexture(attachment.texture), level: attachment.mipLevel,
                     layer: attachment.depthSlice ?? 0,
                     clear: clear, resolve: attachment.resolve)
        }
        if let depth {
            setDepthAttachment(pass, texture: nativeTexture(depth.texture), level: depth.mipLevel,
                               loadAction: depth.loadAction, stencilLoadAction: depth.stencil?.loadAction)
        }
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { throw GPUError.outOfMemory }
        return encoder
    }

    private func setDepthAttachment(
        _ pass: MTLRenderPassDescriptor, texture: any MTLTexture, level: UInt32, layer: UInt32 = 0,
        loadAction: RenderDepthAttachment.LoadAction, stencilLoadAction: RenderStencilAttachmentState.LoadAction?
    ) {
        if loadAction != .unused {
            let attachment = pass.depthAttachment!
            attachment.texture = texture
            attachment.level = Int(level)
            attachment.slice = Int(layer)
            attachment.storeAction = .store
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
            stencil.storeAction = .store
            switch stencilLoadAction {
            case .load, .readOnly: stencil.loadAction = .load
            case .clear(let value):
                stencil.loadAction = .clear
                stencil.clearStencil = value & 0xff
            }
        }
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
private extension RenderColorWriteMask {
    var metal: MTLColorWriteMask {
        var mask: MTLColorWriteMask = []
        if contains(.red) { mask.insert(.red) }
        if contains(.green) { mask.insert(.green) }
        if contains(.blue) { mask.insert(.blue) }
        if contains(.alpha) { mask.insert(.alpha) }
        return mask
    }
}

private extension RenderBlendOperation {
    var metal: MTLBlendOperation {
        switch self {
        case .add: .add
        case .subtract: .subtract
        case .reverseSubtract: .reverseSubtract
        case .min: .min
        case .max: .max
        }
    }
}

private extension RenderBlendFactor {
    var metal: MTLBlendFactor {
        switch self {
        case .zero: .zero
        case .one: .one
        case .sourceColor: .sourceColor
        case .oneMinusSourceColor: .oneMinusSourceColor
        case .destinationColor: .destinationColor
        case .oneMinusDestinationColor: .oneMinusDestinationColor
        case .sourceAlpha: .sourceAlpha
        case .oneMinusSourceAlpha: .oneMinusSourceAlpha
        case .destinationAlpha: .destinationAlpha
        case .oneMinusDestinationAlpha: .oneMinusDestinationAlpha
        case .sourceAlphaSaturated: .sourceAlphaSaturated
        case .constant: .blendColor
        case .oneMinusConstant: .oneMinusBlendColor
        case .source1Color: .source1Color
        case .oneMinusSource1Color: .oneMinusSource1Color
        case .source1Alpha: .source1Alpha
        case .oneMinusSource1Alpha: .oneMinusSource1Alpha
        }
    }
}

private extension RenderPrimitiveState.Topology {
    var metal: (topologyClass: MTLPrimitiveTopologyClass, type: MTLPrimitiveType) {
        switch self {
        case .pointList: (.point, .point)
        case .lineList: (.line, .line)
        case .lineStrip: (.line, .lineStrip)
        case .triangleList: (.triangle, .triangle)
        case .triangleStrip: (.triangle, .triangleStrip)
        }
    }
}

private extension IndexFormat {
    var metal: MTLIndexType { self == .uint16 ? .uint16 : .uint32 }
}

private extension VertexFormat {
    var metal: MTLVertexFormat {
        switch self {
        case .float32: .float
        case .float32x2: .float2
        case .float32x3: .float3
        case .float32x4: .float4
        case .float16: .half
        case .float16x2: .half2
        case .float16x4: .half4
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

private extension RenderStencilFaceState {
    func metal(readMask: UInt32, writeMask: UInt32) -> MTLStencilDescriptor {
        let descriptor = MTLStencilDescriptor()
        descriptor.stencilCompareFunction = compare.metal
        descriptor.stencilFailureOperation = fail.metal
        descriptor.depthFailureOperation = depthFail.metal
        descriptor.depthStencilPassOperation = pass.metal
        descriptor.readMask = readMask & 0xff
        descriptor.writeMask = writeMask & 0xff
        return descriptor
    }
}

private extension RenderStencilOperation {
    var metal: MTLStencilOperation {
        switch self {
        case .keep: .keep
        case .zero: .zero
        case .replace: .replace
        case .invert: .invert
        case .incrementClamp: .incrementClamp
        case .decrementClamp: .decrementClamp
        case .incrementWrap: .incrementWrap
        case .decrementWrap: .decrementWrap
        }
    }
}

private extension CompareFunction {
    var metal: MTLCompareFunction {
        switch self {
        case .never: .never
        case .less: .less
        case .equal: .equal
        case .lessEqual: .lessEqual
        case .greater: .greater
        case .notEqual: .notEqual
        case .greaterEqual: .greaterEqual
        case .always: .always
        }
    }
}

private extension TextureFormat {
    var metal: MTLPixelFormat {
        switch self {
        case .rgba8Unorm: .rgba8Unorm
        case .bgra8Unorm: .bgra8Unorm
        case .r32Uint: .r32Uint
        case .r64Uint: .rg32Uint
        case .stencil8: .stencil8
        case .depth16Unorm: .depth16Unorm
        case .depth24Plus, .depth32Float: .depth32Float
        case .depth24PlusStencil8, .depth32FloatStencil8: .depth32Float_stencil8
        case .rgba8Uint: .rgba8Uint
        case .rgba8Sint: .rgba8Sint
        case .rgba8Snorm: .rgba8Snorm
        case .rgba16Uint: .rgba16Uint
        case .rgba16Sint: .rgba16Sint
        case .rgba16Float: .rgba16Float
        case .r8Unorm: .r8Unorm
        case .r8Snorm: .r8Snorm
        case .r8Uint: .r8Uint
        case .r8Sint: .r8Sint
        case .rg8Unorm: .rg8Unorm
        case .rg8Snorm: .rg8Snorm
        case .rg8Uint: .rg8Uint
        case .rg8Sint: .rg8Sint
        case .rgba8UnormSrgb: .rgba8Unorm_srgb
        case .bgra8UnormSrgb: .bgra8Unorm_srgb
        case .rgb9e5Ufloat: .rgb9e5Float
        case .rgb10a2Unorm: .rgb10a2Unorm
        case .rgb10a2Uint: .rgb10a2Uint
        case .rg11b10Ufloat: .rg11b10Float
        case .r16Unorm: .r16Unorm
        case .rg16Unorm: .rg16Unorm
        case .rgba16Unorm: .rgba16Unorm
        case .r16Snorm: .r16Snorm
        case .rg16Snorm: .rg16Snorm
        case .rgba16Snorm: .rgba16Snorm
        case .r16Uint: .r16Uint
        case .r16Sint: .r16Sint
        case .rg16Uint: .rg16Uint
        case .rg16Sint: .rg16Sint
        case .r16Float: .r16Float
        case .rg16Float: .rg16Float
        case .r32Sint: .r32Sint
        case .r32Float: .r32Float
        case .rg32Uint: .rg32Uint
        case .rg32Sint: .rg32Sint
        case .rg32Float: .rg32Float
        case .rgba32Uint: .rgba32Uint
        case .rgba32Sint: .rgba32Sint
        case .rgba32Float: .rgba32Float
        case .bc1RGBAUnorm: .bc1_rgba
        case .bc1RGBAUnormSrgb: .bc1_rgba_srgb
        case .bc2RGBAUnorm: .bc2_rgba
        case .bc2RGBAUnormSrgb: .bc2_rgba_srgb
        case .bc3RGBAUnorm: .bc3_rgba
        case .bc3RGBAUnormSrgb: .bc3_rgba_srgb
        case .bc4RUnorm: .bc4_rUnorm
        case .bc4RSnorm: .bc4_rSnorm
        case .bc5RGUnorm: .bc5_rgUnorm
        case .bc5RGSnorm: .bc5_rgSnorm
        case .bc6HRGBUfloat: .bc6H_rgbuFloat
        case .bc6HRGBFloat: .bc6H_rgbFloat
        case .bc7RGBAUnorm: .bc7_rgbaUnorm
        case .bc7RGBAUnormSrgb: .bc7_rgbaUnorm_srgb
        case .etc2RGB8Unorm: .etc2_rgb8
        case .etc2RGB8UnormSrgb: .etc2_rgb8_srgb
        case .etc2RGB8A1Unorm: .etc2_rgb8a1
        case .etc2RGB8A1UnormSrgb: .etc2_rgb8a1_srgb
        case .etc2RGBA8Unorm: .eac_rgba8
        case .etc2RGBA8UnormSrgb: .eac_rgba8_srgb
        case .eacR11Unorm: .eac_r11Unorm
        case .eacR11Snorm: .eac_r11Snorm
        case .eacRG11Unorm: .eac_rg11Unorm
        case .eacRG11Snorm: .eac_rg11Snorm
        case .astc4x4Unorm: .astc_4x4_ldr
        case .astc4x4UnormSrgb: .astc_4x4_srgb
        case .astc4x4HDR: .astc_4x4_hdr
        case .astc5x4Unorm: .astc_5x4_ldr
        case .astc5x4UnormSrgb: .astc_5x4_srgb
        case .astc5x4HDR: .astc_5x4_hdr
        case .astc5x5Unorm: .astc_5x5_ldr
        case .astc5x5UnormSrgb: .astc_5x5_srgb
        case .astc5x5HDR: .astc_5x5_hdr
        case .astc6x5Unorm: .astc_6x5_ldr
        case .astc6x5UnormSrgb: .astc_6x5_srgb
        case .astc6x5HDR: .astc_6x5_hdr
        case .astc6x6Unorm: .astc_6x6_ldr
        case .astc6x6UnormSrgb: .astc_6x6_srgb
        case .astc6x6HDR: .astc_6x6_hdr
        case .astc8x5Unorm: .astc_8x5_ldr
        case .astc8x5UnormSrgb: .astc_8x5_srgb
        case .astc8x5HDR: .astc_8x5_hdr
        case .astc8x6Unorm: .astc_8x6_ldr
        case .astc8x6UnormSrgb: .astc_8x6_srgb
        case .astc8x6HDR: .astc_8x6_hdr
        case .astc8x8Unorm: .astc_8x8_ldr
        case .astc8x8UnormSrgb: .astc_8x8_srgb
        case .astc8x8HDR: .astc_8x8_hdr
        case .astc10x5Unorm: .astc_10x5_ldr
        case .astc10x5UnormSrgb: .astc_10x5_srgb
        case .astc10x5HDR: .astc_10x5_hdr
        case .astc10x6Unorm: .astc_10x6_ldr
        case .astc10x6UnormSrgb: .astc_10x6_srgb
        case .astc10x6HDR: .astc_10x6_hdr
        case .astc10x8Unorm: .astc_10x8_ldr
        case .astc10x8UnormSrgb: .astc_10x8_srgb
        case .astc10x8HDR: .astc_10x8_hdr
        case .astc10x10Unorm: .astc_10x10_ldr
        case .astc10x10UnormSrgb: .astc_10x10_srgb
        case .astc10x10HDR: .astc_10x10_hdr
        case .astc12x10Unorm: .astc_12x10_ldr
        case .astc12x10UnormSrgb: .astc_12x10_srgb
        case .astc12x10HDR: .astc_12x10_hdr
        case .astc12x12Unorm: .astc_12x12_ldr
        case .astc12x12UnormSrgb: .astc_12x12_srgb
        case .astc12x12HDR: .astc_12x12_hdr
        }
    }
}
#endif
