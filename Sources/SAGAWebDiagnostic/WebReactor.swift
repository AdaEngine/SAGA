#if os(WASI) && compiler(>=6.3)
@_spi(ExperimentalCustomExecutors) import _Concurrency
import SAGATypes

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

actor BrowserRPC {
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
func start(_ maxBufferSize: UInt64, _ maxTextureDimension2D: UInt32, _ presentation: UInt32, _ readableStorage: UInt32) {
    _Concurrency._createExecutors(factory: ReactorFactory.self)
    Task {
        let status: UInt32
        let message: String
        do {
            let features: Set<GPUFeature> = readableStorage == 0 ? [] : [.storageTextureRead, .storageTextureReadWrite]
            let capabilities = try DeviceCapabilities(maxBufferSize: maxBufferSize, features: features,
                                                       maxTextureDimension2D: maxTextureDimension2D)
            message = try await runWebDiagnostic(capabilities: capabilities, supportsPresentation: presentation != 0)
            status = 0
        } catch {
            message = String(describing: error)
            status = 1
        }
        Array(message.utf8).withUnsafeBufferPointer { finish(status, $0.baseAddress, UInt32($0.count)) }
    }
}
#endif
