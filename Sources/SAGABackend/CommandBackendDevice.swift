import SAGATypes

/// Optional trusted command SPI. Existing browser hosts remain valid without implementing it.
/// Results contain one snapshot per read command, in command order, after real GPU completion.
public protocol CommandBackendDevice: BackendDevice {
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]]
}

/// Single consumer completion latch. Deadlines end waiting, never imply GPU retirement.
public actor BackendCompletion {
    private var result: Result<Void, GPUError>?
    private var waiter: CheckedContinuation<Void, Error>?
    private var waiting = false

    public init() {}

    public func finish(_ result: Result<Void, GPUError>) {
        guard self.result == nil else { return }
        self.result = result
        waiter?.resume(with: result.mapError { $0 as any Error })
        waiter = nil
    }

    public func wait(timeout: Duration) async throws {
        guard !waiting else { throw GPUError.validation("Completion already has a consumer") }
        waiting = true
        if let result { return try result.get() }
        if timeout <= .zero { finish(.failure(.timeout)); throw GPUError.timeout }
        let timer = Task {
            do { try await Task.sleep(for: timeout) }
            catch { return }
            finish(.failure(.timeout))
        }
        defer { timer.cancel() }
        try await withCheckedThrowingContinuation { waiter = $0 }
    }
}
