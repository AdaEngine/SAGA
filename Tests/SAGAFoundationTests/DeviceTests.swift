import Testing
import SAGA
import SAGACore
import SAGATypes
import SAGAWebGPU

private let buffer = BufferDescriptor(size: 16, usage: .storage)

@Test func invalidDescriptorsNeverReachBackend() async throws {
    let backend = TestBackend()
    let device = try await Device.make(backend: backend)
    await #expect(throws: GPUError.self) { try await device.createBuffer(.init(size: 2048, usage: .storage)) }
    await #expect(throws: GPUError.self) { try await device.createBuffer(.init(size: 4, usage: .storage, mappedAtCreation: true)) }
    #expect(await backend.snapshot().isEmpty)
}

@Test func resourceIdentityRejectsForeignForgedAndDestroyedIDs() async throws {
    let backend = TestBackend()
    let first = try await Device.make(backend: backend)
    let second = try await Device.make(backend: backend)
    let id = try await first.createBuffer(buffer)
    await #expect(throws: GPUError.self) { try await second.validateResource(id) }
    await #expect(throws: GPUError.self) { try await first.validateResource(ResourceID(device: first.identity, serial: 999)) }
    await #expect(throws: GPUError.self) { try await second.destroyBuffer(id) }
    try await first.destroyBuffer(id)
    try await first.destroyBuffer(id)
    await #expect(throws: GPUError.self) { try await first.submitResourceUse([id]) }
    let new = try await first.createBuffer(buffer)
    #expect(new != id)
    #expect(await backend.snapshot().filter { $0 == "destroy:1" }.count == 1)
}

@Test func acceptedWorkPinsResourcesAndQueueCompletesInOrder() async throws {
    let gate = Gate()
    let backend = TestBackend(submissionGate: gate)
    let device = try await Device.make(backend: backend)
    let id = try await device.createBuffer(buffer)
    let first = try await device.submitResourceUse([id, id])
    await backend.submissionStarted.wait()
    let second = try await device.submitResourceUse([id])
    try await device.destroyBuffer(id)
    #expect(await backend.snapshot() == ["allocate:1", "begin:1"])
    await gate.release()
    try await first.wait(); try await second.wait()
    #expect(await backend.snapshot() == ["allocate:1", "begin:1", "end:1", "begin:2", "end:2", "destroy:1"])
    #expect(await backend.allocationCount() == 0)
}

@Test func failedSubmissionDoesNotPoisonQueueAndReleasesPins() async throws {
    let backend = TestBackend(failFirstSubmission: true)
    let device = try await Device.make(backend: backend)
    let id = try await device.createBuffer(buffer)
    let first = try await device.submitResourceUse([id])
    let second = try await device.submitResourceUse([id])
    await #expect(throws: GPUError.self) { try await first.wait() }
    try await second.wait()
    try await device.destroyBuffer(id)
    #expect(await backend.snapshot() == ["allocate:1", "begin:1", "begin:2", "end:2", "destroy:1"])
}

@Test func cancelledAllocationRollsBack() async throws {
    let gate = Gate(); let backend = TestBackend(allocationGate: gate)
    let device = try await Device.make(backend: backend)
    let task = Task { try await device.createBuffer(buffer) }
    await backend.allocationStarted.wait()
    task.cancel()
    await gate.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await backend.allocationCount() == 0)
    #expect(await device.takeUncapturedErrors().isEmpty)
}

@Test func cancelledWaiterDoesNotCancelCommittedWork() async throws {
    let gate = Gate(); let backend = TestBackend(submissionGate: gate)
    let device = try await Device.make(backend: backend)
    let id = try await device.createBuffer(buffer)
    let ticket = try await device.submitResourceUse([id])
    await backend.submissionStarted.wait()
    let waiter = Task { try await ticket.wait() }
    waiter.cancel()
    await gate.release()
    await #expect(throws: CancellationError.self) { try await waiter.value }
    try await ticket.wait()
    #expect(await backend.snapshot().contains("end:1"))
}

@Test func deviceLossDuringAllocationCannotPublishResource() async throws {
    let gate = Gate(); let backend = TestBackend(allocationGate: gate)
    let device = try await Device.make(backend: backend)
    let task = Task { try await device.createBuffer(buffer) }
    await backend.allocationStarted.wait()
    await device.lose(reason: "unplugged")
    await gate.release()
    await #expect(throws: GPUError.self) { try await task.value }
    #expect(await backend.allocationCount() == 0)
    await #expect(throws: GPUError.self) { try await device.createBuffer(buffer) }
}

@Test func errorScopesCaptureFirstMatchingErrorAndEnforceLIFO() async throws {
    let device = try await Device.make(backend: TestBackend())
    let outer = try await device.pushErrorScope(.validation)
    let inner = try await device.pushErrorScope(.outOfMemory)
    await #expect(throws: GPUError.self) { try await device.createBuffer(.init(size: 2, usage: [])) }
    await #expect(throws: GPUError.self) { try await device.popErrorScope(outer) }
    #expect(try await device.popErrorScope(inner) == nil)
    #expect(try await device.popErrorScope(outer) != nil)
    #expect(await device.takeUncapturedErrors().isEmpty)
}

@Test func scopePopWaitsForOperationsStartedInsideScope() async throws {
    let gate = Gate()
    let backend = TestBackend(allocationGate: gate, allocationError: .outOfMemory)
    let device = try await Device.make(backend: backend)
    let scope = try await device.pushErrorScope(.outOfMemory)
    let allocation = Task { try await device.createBuffer(buffer) }
    await backend.allocationStarted.wait()
    let pop = Task { try await device.popErrorScope(scope) }
    // Both orderings of pop versus backend release must retain the operation's original scope.
    await gate.release()
    await #expect(throws: GPUError.self) { try await allocation.value }
    #expect(try await pop.value == .outOfMemory)
    #expect(await device.takeUncapturedErrors().isEmpty)
}

@Test func browserBoundaryForwardsOnlyThroughExplicitHost() async throws {
    let host = TestBackend()
    let device = try await Device.make(backend: BrowserBackend(host: host))
    let id = try await device.createBuffer(buffer)
    let ticket = try await device.submitResourceUse([id]); try await ticket.wait()
    try await device.destroyBuffer(id)
    #expect(await host.snapshot() == ["allocate:1", "begin:1", "end:1", "destroy:1"])
}

@Test func closeWaitsForAcceptedWorkAndReleasesAllResources() async throws {
    let gate = Gate(); let backend = TestBackend(submissionGate: gate)
    let device = try await Device.make(backend: backend)
    let first = try await device.createBuffer(buffer)
    _ = try await device.createBuffer(buffer)
    let ticket = try await device.submitResourceUse([first])
    await backend.submissionStarted.wait()
    let closing = Task { await device.close() }
    await gate.release()
    await closing.value
    try await ticket.wait()
    #expect(await backend.allocationCount() == 0)
    #expect(await backend.snapshot().filter { $0.hasPrefix("destroy:") }.count == 2)
    await #expect(throws: GPUError.self) { try await device.submitResourceUse([]) }
    await device.close()
}

@Test func closeWaitsForPendingAllocationRollback() async throws {
    let gate = Gate(); let backend = TestBackend(allocationGate: gate)
    let device = try await Device.make(backend: backend)
    let allocation = Task { try await device.createBuffer(buffer) }
    await backend.allocationStarted.wait()
    await device.lose(reason: "closing")
    let closing = Task { await device.close() }
    await gate.release()
    await closing.value
    await #expect(throws: GPUError.self) { try await allocation.value }
    #expect(await backend.allocationCount() == 0)
}

@Test func cancelledBeforeSubmitNeverEnqueuesWork() async throws {
    let backend = TestBackend(); let device = try await Device.make(backend: backend)
    let id = try await device.createBuffer(buffer)
    let start = Gate()
    let task = Task { await start.wait(); return try await device.submitResourceUse([id]) }
    task.cancel(); await start.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await backend.snapshot() == ["allocate:1"])
    await device.close()
}

@Test func asyncErrorKeepsOriginalScopeAndDoesNotLeakIntoNewScope() async throws {
    let gate = Gate(); let backend = TestBackend(submissionGate: gate, failFirstSubmission: true)
    let device = try await Device.make(backend: backend)
    let old = try await device.pushErrorScope(.internalError)
    let ticket = try await device.submitResourceUse([])
    await backend.submissionStarted.wait()
    let new = try await device.pushErrorScope(.internalError)
    await gate.release()
    await #expect(throws: GPUError.self) { try await ticket.wait() }
    #expect(try await device.popErrorScope(new) == nil)
    #expect(try await device.popErrorScope(old) == .backendFailure("test failure"))
    await device.close()
}

@Test func fullMatchingScopeDoesNotLeakLaterErrorsToOuterScope() async throws {
    let device = try await Device.make(backend: TestBackend())
    let outer = try await device.pushErrorScope(.validation)
    let inner = try await device.pushErrorScope(.validation)
    for _ in 0..<2 {
        await #expect(throws: GPUError.self) { try await device.createBuffer(.init(size: 1, usage: [])) }
    }
    #expect(try await device.popErrorScope(inner) != nil)
    #expect(try await device.popErrorScope(outer) == nil)
    #expect(await device.takeUncapturedErrors().isEmpty)
    await device.close()
}

@Test func backendDeviceLossIsTerminalForAllocationAndQueue() async throws {
    let backend = TestBackend(submissionError: .deviceLost("driver lost"))
    let device = try await Device.make(backend: backend)
    let id = try await device.createBuffer(buffer)
    let ticket = try await device.submitResourceUse([id])
    await #expect(throws: GPUError.deviceLost("driver lost")) { try await ticket.wait() }
    await #expect(throws: GPUError.deviceLost("driver lost")) { try await device.createBuffer(buffer) }
    await #expect(throws: GPUError.deviceLost("driver lost")) { try await device.submitResourceUse([]) }
    await device.close()
    #expect(await backend.allocationCount() == 0)
}

@Test func backendDeviceLossDuringAllocationIsTerminal() async throws {
    let backend = TestBackend(allocationError: .deviceLost("driver lost"))
    let device = try await Device.make(backend: backend)
    await #expect(throws: GPUError.deviceLost("driver lost")) { try await device.createBuffer(buffer) }
    await #expect(throws: GPUError.deviceLost("driver lost")) { try await device.submitResourceUse([]) }
    #expect(await backend.allocationCount() == 0)
    await device.close()
}

@Test(arguments: [false, true])
func overlappingCloseWaitsForWholeTeardownBatch(firstIsLoss: Bool) async throws {
    let backend = TwoDestroyBackend()
    let device = try await Device.make(backend: backend)
    _ = try await device.createBuffer(buffer); _ = try await device.createBuffer(buffer)
    let first = Task {
        if firstIsLoss { await device.lose(reason: "external loss") }
        else { await device.close() }
    }
    await backend.firstStarted.wait()
    let second = Task { await device.close(); return await backend.completedCount() }
    // Let the second caller enter while the first destroy remains held by a deterministic gate.
    for _ in 0..<100 { await Task.yield() }
    await backend.releaseFirst.release()
    await backend.secondStarted.wait()
    for _ in 0..<100 { await Task.yield() }
    await backend.releaseSecond.release()
    await first.value
    #expect(await second.value == 2)
}
