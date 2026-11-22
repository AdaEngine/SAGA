import SAGATypes
import SAGABackend
import SAGAShaderIR
import Foundation
#if canImport(Darwin)
import Darwin
#endif
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

    public static func makeDevice(
        timeout: Duration = .seconds(10), shaderLogHandler: (@Sendable (String) -> Void)? = nil
    ) throws -> MetalDevice {
        try makeDevice(adapter: requestAdapter(), timeout: timeout, shaderLogHandler: shaderLogHandler)
    }

    public static func makeDevice(
        adapter: MetalAdapter?, timeout: Duration = .seconds(10), shaderLogHandler: (@Sendable (String) -> Void)? = nil
    ) throws -> MetalDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        return try MetalDevice(adapter: adapter, timeout: timeout, surface: nil, shaderLogHandler: shaderLogHandler)
    }

    #if canImport(Metal) && canImport(QuartzCore)
    /// Creates a device that can attach Metal layers after GPU resources have been created.
    public static func makePresentableDevice(
        adapter: MetalAdapter? = requestAdapter(), timeout: Duration = .seconds(10)
    ) throws -> MetalDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        return try MetalDevice(adapter: adapter, timeout: timeout, surface: nil, enablePresentation: true, shaderLogHandler: nil)
    }

    @MainActor
    public static func makeDevice(
        surface: MetalPresentationSurface, adapter: MetalAdapter? = requestAdapter(),
        timeout: Duration = .seconds(10), shaderLogHandler: (@Sendable (String) -> Void)? = nil
    ) throws -> MetalDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        let backend = try MetalDevice(adapter: adapter, timeout: timeout, surface: surface, shaderLogHandler: shaderLogHandler)
        try surface.attach(to: adapter.native)
        return backend
    }
    #endif
}

@MainActor
public final class MetalDevice: ComputeAccelerationStructureBackendDevice, PresentationBackendDevice, SamplerBackendDevice, TextureViewBackendDevice,
                                QuerySetBackendDevice, BufferMappingBackendDevice, PlanarExternalTextureBackendDevice, MeshBackendDevice {
    public nonisolated let capabilities: DeviceCapabilities
    public nonisolated let adapterName: String
    public nonisolated let supportsPresentation: Bool
    let timeout: Duration
    var terminal = false
    #if canImport(Metal)
    let native: any MTLDevice
    let queue: any MTLCommandQueue
    var surfaces: [UInt32: MetalPresentationSurface]
    var nextSurfaceID: UInt32 = 0
    let timestampCounters: (any MTLCounterSet)?
    struct Buffer {
        let descriptor: BufferDescriptor
        let native: any MTLBuffer
        var mapping: BufferMapping?
    }
    struct QuerySet {
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
    struct Batch {
        let commandBuffer: any MTLCommandBuffer
        let buffers: [any MTLBuffer]
        let pipelines: [any MTLComputePipelineState]
        let textures: [any MTLTexture]
        let samplers: [any MTLSamplerState]
        let renderPipelines: [RenderState]
        let drawable: (any MTLDrawable)?
        var meshPipelines: [any MTLRenderPipelineState] = []
        var counterSampleBuffers: [any MTLCounterSampleBuffer] = []
        var committedHandlers: [@MainActor @Sendable () -> Void] = []
        var accelerationBuilds: [ResourceID: PendingAccelerationBuild] = [:]
        var retainedAccelerationBuilds: [MetalAccelerationStructureBuild] = []
        var groupingStats: GroupingStats?
    }
    struct RenderState {
        let pipeline: any MTLRenderPipelineState
        let depth: (any MTLDepthStencilState)?
        let writesDepth: Bool
        let usesVertexPulling: Bool
        let vertexRuntimeLengths: [(ComputeBindingSlot, UInt32, UInt32)]
        let fragmentRuntimeLengths: [(ComputeBindingSlot, UInt32, UInt32)]
        let vertexExternalTextures: [ComputeShaderArtifact.ExternalTextureLayout]
        let fragmentExternalTextures: [ComputeShaderArtifact.ExternalTextureLayout]
        let fragmentTextureAtomicGuardBufferIndex: UInt32?
        let vertexImmediateBufferIndex: UInt32?
        let fragmentImmediateBufferIndex: UInt32?
        var metalRenderTargetLayer: UInt32? = nil
    }

    static let traceGrouping = ProcessInfo.processInfo.environment["ADAENGINE_SAGA_TRACE_GROUPING"] == "1"
    struct GroupingStats {
        var indexedDraws = 0
        var renderEncoders = 0
        var indexEncoders = 0
        var groups = 0
        var groupedDraws = 0
        var cpu: UInt64 = 0
    }

    struct TextureSubresource: Hashable {
        let level: UInt32
        let layer: UInt32
    }

    struct Texture {
        let descriptor: TextureDescriptor
        let native: any MTLTexture
        var initializedSubresources: Set<TextureSubresource> = []
    }
    struct Sampler {
        let native: any MTLSamplerState
        let descriptor: SamplerDescriptor
    }

    struct TextureView {
        let resource: TextureViewResource
        let native: any MTLTexture
    }
    struct ExternalTexture {
        let planes: [ResourceID]
        let parameters: [UInt32]
    }
    struct Pipeline {
        let state: any MTLComputePipelineState
        let threadsPerThreadgroup: MTLSize
        let bufferIndices: [ComputeBindingSlot: Int]
        let accelerationIndices: [ComputeBindingSlot: Int]
        let textureIndices: [ComputeBindingSlot: Int]
        let samplerIndices: [ComputeBindingSlot: Int]
        let runtimeLengths: [(slot: ComputeBindingSlot, stride: UInt32, offset: UInt32)]
        let lengthBufferIndex: UInt32?
        let immediateBufferIndex: UInt32?
        let externalTextures: [ComputeShaderArtifact.ExternalTextureLayout]
    }
    struct PendingAccelerationBuild {
        let serial: UInt64
        let build: MetalAccelerationStructureBuild
    }
    struct AccelerationStructure {
        let descriptor: AccelerationStructureDescriptor
        var build: PendingAccelerationBuild?
    }
    var nextAccelerationBuild: UInt64 = 0
    var accelerationStructures: [ResourceID: AccelerationStructure] = [:]
    var buffers: [ResourceID: Buffer] = [:]
    var querySets: [ResourceID: QuerySet] = [:]
    var pipelines: [ResourceID: any MTLComputePipelineState] = [:]
    var computePipelines: [ResourceID: Pipeline] = [:]
    var textures: [ResourceID: Texture] = [:]
    var samplers: [ResourceID: Sampler] = [:]
    var textureViews: [ResourceID: TextureView] = [:]
    var externalTextures: [ResourceID: ExternalTexture] = [:]
    var renderPipelines: [ResourceID: RenderState] = [:]
    struct MeshState {
        let pipeline: any MTLRenderPipelineState
        let depth: (any MTLDepthStencilState)?
        let writesDepth: Bool
        let objectThreads: MTLSize
        let meshThreads: MTLSize
        let objectUniforms: [(ComputeBindingSlot, Int)]
        let objectStorage: [(ComputeBindingSlot, Int)]
        let meshUniforms: [(ComputeBindingSlot, Int)]
        let meshStorage: [(ComputeBindingSlot, Int)]
        let objectTextures: [(ComputeBindingSlot, Int)]
        let meshTextures: [(ComputeBindingSlot, Int)]
        let objectSamplers: [(ComputeBindingSlot, Int)]
        let meshSamplers: [(ComputeBindingSlot, Int)]
        let fragmentBuffers: [(ComputeBindingSlot, Int)]
        let fragmentTextures: [(ComputeBindingSlot, Int)]
        let fragmentSamplers: [(ComputeBindingSlot, Int)]
        let fragmentLengths: [(ComputeBindingSlot, UInt32, UInt32)]
        let fragmentTextureAtomicGuardBufferIndex: UInt32?
        let objectLengths: [(ComputeBindingSlot, UInt32, UInt32)]
        let meshLengths: [(ComputeBindingSlot, UInt32, UInt32)]
    }
    var meshPipelines: [ResourceID: MeshState] = [:]
    var surfacePipeline: (any MTLRenderPipelineState)?
    var presentingSurface: UInt32?
    var surfaceRetirement: CheckedContinuation<Void, Never>?
    var indexValidator: MetalIndexValidator?
    var dispatchValidator: MetalDispatchValidator?
    var meshDispatchValidator: MetalDispatchValidator?
    var indirectDrawValidator: MetalIndirectDrawValidator?
    var timestampMarkerPipeline: (any MTLComputePipelineState)?
    var cachedDiagnosticPipeline: (any MTLComputePipelineState)?
    var retirementHook: (@Sendable () async -> Void)?
    var retirementWaiters: [CheckedContinuation<Void, Never>] = []
    var nextPipeline: UInt64 = 0
    var nextBatch: UInt64 = 0
    var inFlight: [UInt64: Batch] = [:]
    #endif

    nonisolated fileprivate init(
        adapter: MetalAdapter, timeout: Duration, surface: MetalPresentationSurface?,
        enablePresentation: Bool = false,
        shaderLogHandler: (@Sendable (String) -> Void)?
    ) throws {
        guard timeout > .zero else { throw GPUError.validation("Timeout must be positive") }
        self.timeout = timeout
        adapterName = adapter.name
        supportsPresentation = enablePresentation || surface != nil
        #if canImport(Metal)
        native = adapter.native
        self.surfaces = surface.map { [0: $0] } ?? [:]
        let supportsShaderLogging = if #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) {
            native.supportsFamily(.apple6) || native.supportsFamily(.mac2)
        } else { false }
        let commandQueue: (any MTLCommandQueue)?
        if let shaderLogHandler {
            guard supportsShaderLogging, #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) else {
                throw GPUError.unsupported("Metal shader logging requires MSL 3.2 and a supported GPU")
            }
            let logging = MTLLogStateDescriptor()
            logging.level = .info
            let state: any MTLLogState
            do { state = try native.makeLogState(descriptor: logging) }
            catch { throw GPUError.backendFailure("Unable to create Metal shader log state") }
            state.addLogHandler { _, _, _, message in shaderLogHandler(message as String) }
            let descriptor = MTLCommandQueueDescriptor()
            descriptor.logState = state
            commandQueue = native.makeCommandQueue(descriptor: descriptor)
        } else {
            commandQueue = native.makeCommandQueue()
        }
        guard let commandQueue else { throw GPUError.outOfMemory }
        self.queue = commandQueue
        // MSL short and ushort are available on every Metal device.
        var features: Set<GPUFeature> = [.storageTextureRead, .shaderF16, .shaderInt16, .indirectFirstInstance, .clipDistances,
                                         .shaderEarlyDepthTest, .linearIndexing, .subgroupUniformity]
        if #available(macOS 13, iOS 16, *), native.supportsFamily(.metal3) { features.insert(.meshShaders) }
        if native.supportsRaytracing { features.insert(.rayQuery) }
        if supportsShaderLogging { features.insert(.shaderDebugPrintf) }
        var maxMultiviewViewCount: UInt32 = 1
        while maxMultiviewViewCount < 32 && native.supportsVertexAmplificationCount(Int(maxMultiviewViewCount * 2)) {
            maxMultiviewViewCount *= 2
        }
        if maxMultiviewViewCount > 1 { features.formUnion([.multiview, .selectiveMultiview]) }
        if native.supportsRasterizationRateMap(layerCount: 1) { features.insert(.rasterizationRateMap) }
        if native.supportsTextureSampleCount(4) && (native.supportsFamily(.apple7) || native.supportsFamily(.mac2)) {
            features.insert(.multisampleArray)
        }
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
        features.formUnion([.depth32FloatStencil8, .bgra8UnormStorage, .vertexFormatSnorm1010102, .vertexWritableStorage])
        if #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) { features.insert(.memoryDecorationCoherent) }
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
        if adapter.native.supportsFamily(.apple7) || adapter.native.supportsFamily(.mac2) {
            features.insert(.cooperativeMatrix)
        }
        if adapter.native.supportsShaderBarycentricCoordinates { features.insert(.shaderBarycentrics) }
        if #available(macOS 26, iOS 26, tvOS 26, visionOS 26, *), adapter.native.supportsFamily(.apple10) {
            features.insert(.shaderPerVertex)
        }
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
        let nativeThreads = adapter.native.maxThreadsPerThreadgroup
        let workgroupSize = DispatchWorkgroups(x: UInt32(clamping: nativeThreads.width),
            y: UInt32(clamping: nativeThreads.height), z: UInt32(clamping: nativeThreads.depth))
        #if os(macOS)
        let legacyInterstageVariables: UInt32 = 30
        #else
        let legacyInterstageVariables: UInt32 = 15
        #endif
        let maxInterstageVariables = adapter.native.supportsFamily(.apple4) ? 31 : legacyInterstageVariables
        let colorAttachmentBytes: UInt32
        if adapter.native.supportsFamily(.apple7) {
            colorAttachmentBytes = 128
        } else if adapter.native.supportsFamily(.apple4) {
            colorAttachmentBytes = 64
        } else if adapter.native.supportsFamily(.apple2) {
            colorAttachmentBytes = 32
        } else {
            #if os(macOS)
            colorAttachmentBytes = 128
            #else
            colorAttachmentBytes = 16
            #endif
        }
        #if os(macOS)
        let colorAttachmentSlots = RenderStageArtifact.nativeColorAttachmentCapacity
        #else
        let colorAttachmentSlots = adapter.native.supportsFamily(.apple2) ? RenderStageArtifact.nativeColorAttachmentCapacity : 4
        #endif
        var cooperativeProperties: [CooperativeMatrixProperties] = []
        if features.contains(.cooperativeMatrix) {
            for m: UInt32 in [8, 16] {
                for n: UInt32 in [8, 16] {
                    for k: UInt32 in [8, 16] {
                        cooperativeProperties.append(.init(mSize: m, nSize: n, kSize: k,
                            inputType: .float32, accumulatorType: .float32))
                        cooperativeProperties.append(.init(mSize: m, nSize: n, kSize: k,
                            inputType: .float16, accumulatorType: .float16))
                        cooperativeProperties.append(.init(mSize: m, nSize: n, kSize: k,
                            inputType: .float16, accumulatorType: .float32))
                    }
                }
            }
        }
        var taskWorkgroupCount: UInt32 = 0
        var meshWorkgroupCount: UInt32 = 0
        if features.contains(.meshShaders) {
            var metal4 = false
            #if !targetEnvironment(simulator)
            if #available(macOS 26, iOS 26, tvOS 26, visionOS 26, *) {
                metal4 = native.supportsFamily(.metal4)
            }
            #endif
            taskWorkgroupCount = metal4 || native.supportsFamily(.apple7) ? UInt32.max : 1024
            if #available(macOS 26, iOS 26, tvOS 26, visionOS 26, *), native.supportsFamily(.apple10) {
                meshWorkgroupCount = 1 << 22
            } else if native.supportsFamily(.apple9) {
                meshWorkgroupCount = 1 << 20
            } else {
                meshWorkgroupCount = 1024
            }
        }
        capabilities = try DeviceCapabilities(maxBufferSize: UInt64(adapter.native.maxBufferLength), features: features,
                                             timestampPeriodNanoseconds: period, multisampledFloat32Formats: float32MSAA,
                                             maxImmediateSize: 4096,
                                             maxComputeWorkgroupStorageSize: UInt32(clamping: adapter.native.maxThreadgroupMemoryLength),
                                             maxComputeInvocationsPerWorkgroup: max(workgroupSize.x, workgroupSize.y, workgroupSize.z),
                                             maxComputeWorkgroupSize: workgroupSize,
                                             maxVertexAttributes: RenderStageArtifact.metalVertexAttributeCapacity,
                                             maxInterStageShaderVariables: maxInterstageVariables,
                                             maxColorAttachments: colorAttachmentSlots,
                                             maxColorAttachmentBytesPerSample: colorAttachmentBytes,
                                             maxMultiviewViewCount: maxMultiviewViewCount > 1 ? maxMultiviewViewCount : 0,
                                             cooperativeMatrixProperties: cooperativeProperties,
                                             maxTaskWorkgroupTotalCount: taskWorkgroupCount,
                                             maxTaskWorkgroupsPerDimension: taskWorkgroupCount,
                                             maxMeshWorkgroupTotalCount: meshWorkgroupCount,
                                             maxMeshWorkgroupsPerDimension: meshWorkgroupCount,
                                             maxTaskInvocationsPerWorkgroup: features.contains(.meshShaders) ? 1024 : 0,
                                             maxTaskInvocationsPerDimension: features.contains(.meshShaders) ? 1024 : 0,
                                             maxMeshInvocationsPerWorkgroup: features.contains(.meshShaders) ? 1024 : 0,
                                             maxMeshInvocationsPerDimension: features.contains(.meshShaders) ? 1024 : 0,
                                             maxTaskPayloadSize: features.contains(.meshShaders) ? 16_352 : 0,
                                             maxMeshOutputVertices: 256, maxMeshOutputPrimitives: 256)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    func requireActive() throws {
        guard !terminal else { throw GPUError.deviceLost("Metal device is terminal after command failure/timeout") }
    }
}
