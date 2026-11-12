#if os(WASI) && compiler(>=6.3)
@_spi(ExperimentalCustomExecutors) import _Concurrency
import SAGATypes
import SAGAWebHost

@_extern(wasm, module: "saga_host", name: "schedule")
private func schedule(_ pointer: UnsafeMutableRawPointer)
@_extern(wasm, module: "saga_host", name: "request")
private func request(_ id: UInt32, _ bytes: UnsafePointer<UInt8>?, _ count: UInt32)
@_extern(wasm, module: "saga_host", name: "copyReply")
private func copyReply(_ id: UInt32, _ bytes: UnsafeMutablePointer<UInt8>?, _ count: UInt32)
@_extern(wasm, module: "saga_host", name: "finish")
private func finish(_ status: UInt32, _ bytes: UnsafePointer<UInt8>?, _ count: UInt32)

private final class Work: Sendable {
    let run: @Sendable () -> Void
    init(_ run: @escaping @Sendable () -> Void) { self.run = run }
}

private final class ReactorExecutor: MainExecutor, TaskExecutor {
    let main: Bool
    init(main: Bool) { self.main = main }

    func enqueue(_ job: consuming ExecutorJob) {
        let job = UnownedJob(job)
        let work = Work {
            if self.main { job.runSynchronously(on: self.asUnownedSerialExecutor()) }
            else { job.runSynchronously(on: self.asUnownedTaskExecutor()) }
        }
        schedule(Unmanaged.passRetained(work).toOpaque())
    }
    func run() throws {}
    func stop() {}
}

private enum ReactorFactory: ExecutorFactory {
    static let mainExecutor: any MainExecutor = ReactorExecutor(main: true)
    static let defaultExecutor: any TaskExecutor = ReactorExecutor(main: false)
}

actor BrowserRPC: BrowserTransport {
    static let shared = BrowserRPC()
    static let packetLimit = 16 * 1024 * 1024
    private var nextID: UInt32 = 0
    private var pending: [UInt32: CheckedContinuation<[UInt8], Error>] = [:]

    func send(_ bytes: [UInt8]) async throws -> [UInt8] {
        guard nextID < UInt32.max, bytes.count <= Self.packetLimit else {
            throw GPUError.validation("Browser bridge packet or request ID exceeds its limit")
        }
        nextID += 1
        let id = nextID
        return try await withCheckedThrowingContinuation { continuation in
            pending[id] = continuation
            bytes.withUnsafeBufferPointer { request(id, $0.baseAddress, UInt32($0.count)) }
        }
    }

    func complete(_ id: UInt32, status: UInt32, bytes: [UInt8]) {
        guard let continuation = pending.removeValue(forKey: id) else { return }
        if status == 0 {
            continuation.resume(returning: bytes)
            return
        }
        let message = String(decoding: bytes, as: UTF8.self)
        let error: GPUError
        switch status {
        case 1: error = .validation(message)
        case 2: error = .outOfMemory
        case 3: error = .deviceLost(message)
        case 4: error = .timeout
        default: error = .backendFailure(message)
        }
        continuation.resume(throwing: error)
    }
}

@_expose(wasm, "saga_run_job")
func runJob(_ pointer: UnsafeMutableRawPointer) {
    Unmanaged<Work>.fromOpaque(pointer).takeRetainedValue().run()
}

@_expose(wasm, "saga_reply")
func reply(_ id: UInt32, _ status: UInt32, _ count: UInt32) {
    guard count <= BrowserRPC.packetLimit else {
        Task { await BrowserRPC.shared.complete(id, status: 5, bytes: Array("Oversized reply".utf8)) }
        return
    }
    var bytes = [UInt8](repeating: 0, count: Int(count))
    bytes.withUnsafeMutableBufferPointer { copyReply(id, $0.baseAddress, count) }
    Task { await BrowserRPC.shared.complete(id, status: status, bytes: bytes) }
}

@_expose(wasm, "saga_start")
func start(_ maxBufferSize: UInt64, _ maxTextureDimension2D: UInt32, _ presentation: UInt32,
           _ readableStorage: UInt32, _ shaderF16: UInt32, _ uniformBufferStandardLayout: UInt32,
           _ indirectFirstInstance: UInt32, _ clipDistances: UInt32, _ primitiveIndex: UInt32, _ depthClipControl: UInt32,
           _ timestampQuery: UInt32, _ subgroups: UInt32, _ subgroupID: UInt32, _ dualSourceBlending: UInt32,
           _ textureCompressionBC: UInt32, _ textureCompressionETC2: UInt32, _ textureCompressionASTC: UInt32,
           _ textureCompressionBCSliced3D: UInt32, _ textureCompressionASTCSliced3D: UInt32, _ float32Filterable: UInt32, _ float32Blendable: UInt32,
           _ multisampledFloat32Mask: UInt32, _ textureFormatsTier1: UInt32, _ textureFormatsTier2: UInt32, _ rg11b10UfloatRenderable: UInt32, _ depth32FloatStencil8: UInt32, _ bgra8UnormStorage: UInt32,
           _ firstVideoFrame: UInt64, _ secondVideoFrame: UInt64, _ closedVideoFrame: UInt64, _ maxImmediateSize: UInt32,
           _ linearIndexing: UInt32, _ subgroupSizeControl: UInt32, _ subgroupMinSize: UInt32, _ subgroupMaxSize: UInt32,
           _ subgroupUniformity: UInt32, _ maxComputeWorkgroupStorageSize: UInt32,
           _ maxComputeInvocationsPerWorkgroup: UInt32, _ maxComputeWorkgroupSizeX: UInt32,
           _ maxComputeWorkgroupSizeY: UInt32, _ maxComputeWorkgroupSizeZ: UInt32,
           _ maxComputeWorkgroupsPerDimension: UInt32, _ maxVertexAttributes: UInt32,
           _ maxVertexBuffers: UInt32, _ maxVertexBufferArrayStride: UInt32, _ vertexFormatSnorm1010102: UInt32,
           _ maxInterStageShaderVariables: UInt32, _ maxColorAttachments: UInt32, _ maxColorAttachmentBytesPerSample: UInt32,
           _ maxBindGroupsPlusVertexBuffers: UInt32) {
    _Concurrency._createExecutors(factory: ReactorFactory.self)
    Task {
        let status: UInt32
        let message: String
        do {
            var parameters = WebHostParameters()
            parameters.maxBufferSize = maxBufferSize
            parameters.maxTextureDimension2D = maxTextureDimension2D
            parameters.presentation = presentation
            parameters.readableStorage = readableStorage
            parameters.shaderF16 = shaderF16
            parameters.uniformBufferStandardLayout = uniformBufferStandardLayout
            parameters.indirectFirstInstance = indirectFirstInstance
            parameters.clipDistances = clipDistances
            parameters.primitiveIndex = primitiveIndex
            parameters.depthClipControl = depthClipControl
            parameters.timestampQuery = timestampQuery
            parameters.subgroups = subgroups
            parameters.subgroupID = subgroupID
            parameters.dualSourceBlending = dualSourceBlending
            parameters.textureCompressionBC = textureCompressionBC
            parameters.textureCompressionETC2 = textureCompressionETC2
            parameters.textureCompressionASTC = textureCompressionASTC
            parameters.textureCompressionBCSliced3D = textureCompressionBCSliced3D
            parameters.textureCompressionASTCSliced3D = textureCompressionASTCSliced3D
            parameters.float32Filterable = float32Filterable
            parameters.float32Blendable = float32Blendable
            parameters.multisampledFloat32Mask = multisampledFloat32Mask
            parameters.textureFormatsTier1 = textureFormatsTier1
            parameters.textureFormatsTier2 = textureFormatsTier2
            parameters.rg11b10UfloatRenderable = rg11b10UfloatRenderable
            parameters.depth32FloatStencil8 = depth32FloatStencil8
            parameters.bgra8UnormStorage = bgra8UnormStorage
            parameters.maxImmediateSize = maxImmediateSize
            parameters.linearIndexing = linearIndexing
            parameters.subgroupSizeControl = subgroupSizeControl
            parameters.subgroupMinSize = subgroupMinSize
            parameters.subgroupMaxSize = subgroupMaxSize
            parameters.subgroupUniformity = subgroupUniformity
            parameters.maxComputeWorkgroupStorageSize = maxComputeWorkgroupStorageSize
            parameters.maxComputeInvocationsPerWorkgroup = maxComputeInvocationsPerWorkgroup
            parameters.maxComputeWorkgroupSizeX = maxComputeWorkgroupSizeX
            parameters.maxComputeWorkgroupSizeY = maxComputeWorkgroupSizeY
            parameters.maxComputeWorkgroupSizeZ = maxComputeWorkgroupSizeZ
            parameters.maxComputeWorkgroupsPerDimension = maxComputeWorkgroupsPerDimension
            parameters.maxVertexAttributes = maxVertexAttributes
            parameters.maxVertexBuffers = maxVertexBuffers
            parameters.maxVertexBufferArrayStride = maxVertexBufferArrayStride
            parameters.vertexFormatSnorm1010102 = vertexFormatSnorm1010102
            parameters.maxInterStageShaderVariables = maxInterStageShaderVariables
            parameters.maxColorAttachments = maxColorAttachments
            parameters.maxColorAttachmentBytesPerSample = maxColorAttachmentBytesPerSample
            parameters.maxBindGroupsPlusVertexBuffers = maxBindGroupsPlusVertexBuffers
            let capabilities = try makeWebDeviceCapabilities(parameters)
            message = try await runWebDiagnostic(capabilities: capabilities, supportsPresentation: presentation != 0,
                                                  videoFrames: (firstVideoFrame, secondVideoFrame, closedVideoFrame))
            status = 0
        } catch {
            message = String(describing: error)
            status = 1
        }
        Array(message.utf8).withUnsafeBufferPointer { finish(status, $0.baseAddress, UInt32($0.count)) }
    }
}
#endif
