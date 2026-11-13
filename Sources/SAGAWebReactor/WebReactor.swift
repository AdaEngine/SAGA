#if os(WASI) && compiler(>=6.3)
@_spi(ExperimentalCustomExecutors) import _Concurrency
import SAGABackend
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

private actor BrowserRPC: BrowserTransport {
    static let shared = BrowserRPC()
    static let packetLimit = browserPacketLimit
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

/// Runs a job scheduled by the reactor. The host executable exports this as `saga_run_job`.
public func sagaWebRunJob(_ pointer: UnsafeMutableRawPointer) {
    Unmanaged<Work>.fromOpaque(pointer).takeRetainedValue().run()
}

/// Delivers a bridge reply. The host executable exports this as `saga_reply`.
public func sagaWebReply(_ id: UInt32, _ status: UInt32, _ count: UInt32) {
    guard count <= BrowserRPC.packetLimit else {
        Task { await BrowserRPC.shared.complete(id, status: 5, bytes: Array("Oversized reply".utf8)) }
        return
    }
    var bytes = [UInt8](repeating: 0, count: Int(count))
    bytes.withUnsafeMutableBufferPointer { copyReply(id, $0.baseAddress, count) }
    Task { await BrowserRPC.shared.complete(id, status: status, bytes: bytes) }
}

/// Starts the reactor and runs `run` with the browser capabilities. Exported by the host as `saga_start`.
public func sagaWebStart(
    _ parameters: WebHostParameters,
    run: @escaping @Sendable (DeviceCapabilities, Bool, any BrowserTransport) async throws -> String
) {
    _Concurrency._createExecutors(factory: ReactorFactory.self)
    let p = parameters
    Task {
        let status: UInt32
        let message: String
        do {
            let capabilities = try makeWebDeviceCapabilities(p)
            message = try await run(capabilities, p.presentation != 0, BrowserRPC.shared)
            status = 0
        } catch {
            message = String(describing: error)
            status = 1
        }
        Array(message.utf8).withUnsafeBufferPointer { finish(status, $0.baseAddress, UInt32($0.count)) }
    }
}
#endif
