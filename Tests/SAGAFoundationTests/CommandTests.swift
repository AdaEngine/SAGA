import Testing
import SAGATypes
import SAGABackend
import SAGACore

actor CommandTestBackend: CommandBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    let started = Gate()
    let gate: Gate?
    let error: GPUError?
    var calls = 0
    var destroyed: [ResourceID] = []
    init(gate: Gate? = nil, error: GPUError? = nil) { self.gate = gate; self.error = error }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) {}
    func destroyBuffer(id: ResourceID) { destroyed.append(id) }
    func submit(_ submission: Submission) {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        calls += 1
        await started.release()
        if let gate { await gate.wait() }
        if let error { throw error }
        return []
    }
    func snapshot() -> (Int, [ResourceID]) { (calls, destroyed) }
}

@Test func commandValidationRejectsSizeUsageAlignmentAndForeignIDsBeforeBackend() async throws {
    let backend = CommandTestBackend()
    let device = try await Device.make(backend: backend)
    let a = try await device.createBuffer(.init(size: 8, usage: [.storage, .copyDestination]))
    let b = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
    let r = try await device.createBuffer(.init(size: 8, usage: [.mapRead, .copyDestination]))
    let p = ResourceID(device: device.identity, serial: 1)
    let invalid: [BufferCommand] = [
        .write(buffer: b, offset: 0, words: [1]),
        .write(buffer: a, offset: UInt64.max, words: [1]),
        .write(buffer: a, offset: 2, words: [1]),
        .write(buffer: a, offset: 4, words: [1, 2]),
        .dispatch(pipeline: p, input: a, output: b, count: 0),
        .dispatch(pipeline: p, input: a, output: b, count: UInt32.max),
        .dispatch(pipeline: p, input: a, output: a, count: 1),
        .dispatch(pipeline: .init(device: device.identity + 1, serial: 1), input: a, output: b, count: 1),
        .copy(source: a, destination: r, byteCount: 4),
        .copy(source: b, destination: r, byteCount: 3),
        .copy(source: b, destination: r, byteCount: 12),
        .copy(source: b, destination: b, byteCount: 4),
        .read(buffer: b, byteCount: 4),
        .read(buffer: r, byteCount: 0),
        .read(buffer: r, byteCount: UInt64.max),
        .read(buffer: .init(device: device.identity + 1, serial: 1), byteCount: 4),
    ]
    for command in invalid {
        await #expect(throws: GPUError.self) { try await device.submitCommands([command]) }
    }
    #expect(await backend.snapshot().0 == 0)
    let valid = try await device.submitCommands([.copy(source: b, destination: r, byteCount: 8)])
    #expect(try await valid.wait().isEmpty)
    #expect(await backend.snapshot().0 == 1)
    await device.close()
}

@Test func commandBuffersStayPinnedAfterDestroyAndWaiterCancellation() async throws {
    let gate = Gate()
    let backend = CommandTestBackend(gate: gate)
    let device = try await Device.make(backend: backend)
    let a = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    let ticket = try await device.submitCommands([.read(buffer: a, byteCount: 4)])
    await backend.started.wait()
    try await device.destroyBuffer(a)
    #expect(await backend.snapshot().1.isEmpty)
    let start = Gate()
    let waiter = Task { await start.wait(); return try await ticket.wait() }
    waiter.cancel(); await start.release()
    await #expect(throws: CancellationError.self) { try await waiter.value }
    #expect(await backend.snapshot().1.isEmpty)
    await gate.release()
    _ = try await ticket.wait()
    #expect(await backend.snapshot().1 == [a])
    await device.close()
}

@Test func commandTimeoutMakesCoreTerminalAndCloseDoesNotHang() async throws {
    let backend = CommandTestBackend(error: .timeout)
    let device = try await Device.make(backend: backend)
    let ticket = try await device.submitCommands([])
    await #expect(throws: GPUError.timeout) { try await ticket.wait() }
    await #expect(throws: GPUError.deviceLost("GPU completion timeout")) {
        try await device.createBuffer(.init(size: 4, usage: .copySource))
    }
    await device.close()
}

@Test func legacyBrowserHostRejectsCommandsInsteadOfFallingBack() async throws {
    let device = try await Device.make(backend: TestBackend())
    await #expect(throws: GPUError.unsupported("Backend does not implement command submission")) {
        try await device.submitCommands([])
    }
    await device.close()
}
