import Testing
import SAGATypes
import SAGABackend
import SAGACore

private actor RebuildLifetimeBackend: AccelerationStructureBackendDevice, CommandBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    private var gate = false
    private var shouldFail = false
    private var started = false
    private var continuation: CheckedContinuation<Void, Never>?
    private(set) var destroyed: [ResourceID] = []
    private(set) var destroyedBuffers: [ResourceID] = []

    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async { destroyedBuffers.append(id) }
    func submit(_ submission: Submission) async throws {}
    func allocateAccelerationStructure(id: ResourceID, descriptor: AccelerationStructureDescriptor) async throws {}
    func destroyAccelerationStructure(id: ResourceID) async { destroyed.append(id) }
    func arm(fail: Bool) { gate = true; shouldFail = fail; started = false }
    func waitForStart() async { while !started { await Task.yield() } }
    func release() { continuation?.resume(); continuation = nil }
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        if gate {
            gate = false
            started = true
            await withCheckedContinuation { continuation = $0 }
            if shouldFail { throw GPUError.validation("injected rebuild failure") }
        }
        return []
    }
}

private func prepareRebuild(_ device: Device) async throws -> (ResourceID, ResourceID, ResourceID) {
    let vertex = try await device.createBuffer(.init(size: 36, usage: .vertex))
    let a = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let b = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertex, vertexStride: 12, vertexCount: 3)
    let ticket = try await device.submitCommands([
        .buildAccelerationStructure(destination: a, geometry: geometry),
        .buildAccelerationStructure(destination: b, geometry: geometry),
        .buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: a)]),
    ])
    _ = try await ticket.wait()
    return (a, b, top)
}

@Test func successfulTopLevelRebuildReleasesOldDependenciesAfterCompletion() async throws {
    let backend = RebuildLifetimeBackend()
    let device = try await Device.make(backend: backend)
    let (a, b, top) = try await prepareRebuild(device)
    await backend.arm(fail: false)
    let rebuild = try await device.submitCommands([
        .buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: b)])
    ])
    await backend.waitForStart()
    try await device.destroyAccelerationStructure(a)
    try await device.destroyAccelerationStructure(b)
    #expect(!(await backend.destroyed).contains(a))
    #expect(!(await backend.destroyed).contains(b))
    await backend.release()
    _ = try await rebuild.wait()
    #expect((await backend.destroyed).contains(a))
    #expect(!(await backend.destroyed).contains(b))
    try await device.destroyAccelerationStructure(top)
    #expect((await backend.destroyed).contains(b))
    await device.close()
}

@Test func failedTopLevelRebuildReleasesNewDependencies() async throws {
    let backend = RebuildLifetimeBackend()
    let device = try await Device.make(backend: backend)
    let (a, b, top) = try await prepareRebuild(device)
    await backend.arm(fail: true)
    let rebuild = try await device.submitCommands([
        .buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: b)])
    ])
    await backend.waitForStart()
    try await device.destroyAccelerationStructure(b)
    #expect(!(await backend.destroyed).contains(b))
    await backend.release()
    await #expect(throws: GPUError.self) { try await rebuild.wait() }
    #expect((await backend.destroyed).contains(b))
    try await device.destroyAccelerationStructure(a)
    try await device.destroyAccelerationStructure(top)
    await device.close()
}

@Test func indexedAccelerationBuildPinsIndexBufferUntilRetirement() async throws {
    let backend = RebuildLifetimeBackend()
    let device = try await Device.make(backend: backend)
    let vertex = try await device.createBuffer(.init(size: 36, usage: .vertex))
    let index = try await device.createBuffer(.init(size: 6, usage: .index))
    let bottom = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertex, vertexStride: 12, vertexCount: 3,
        indexBuffer: .init(buffer: index, format: .uint16), indexCount: 3)
    await backend.arm(fail: false)
    let ticket = try await device.submitCommands([.buildAccelerationStructure(destination: bottom, geometry: geometry)])
    await backend.waitForStart()
    try await device.destroyBuffer(index)
    #expect(!(await backend.destroyedBuffers).contains(index))
    await backend.release()
    _ = try await ticket.wait()
    #expect((await backend.destroyedBuffers).contains(index))
    await device.close()
}
