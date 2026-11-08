import Testing
import SAGATypes
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGAWebGPU
import SAGAMetal

private actor MappingTestHost: BrowserGPUComputeHost, BufferMappingBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    let submitted = Gate()
    let completeSubmission = Gate()
    let opening = Gate()
    let completeOpening = Gate()
    let writing = Gate()
    let completeWriting: Gate?
    private(set) var events: [String] = []

    init(blockWrites: Bool = false) { completeWriting = blockWrites ? Gate() : nil }

    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) {}
    func destroyBuffer(id: ResourceID) { events.append("destroy") }
    func submit(_ submission: Submission) async {
        await submitted.release()
        await completeSubmission.wait()
    }
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) -> [[UInt32]] { [] }
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) {}
    func destroyComputePipeline(id: ResourceID) {}

    func beginBufferMapping(id: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64) async {
        events.append("map")
        await opening.release()
        await completeOpening.wait()
    }
    func readMappedBytes(id: ResourceID, offset: UInt64, size: UInt64) -> [UInt8] {
        Array(repeating: 42, count: Int(size))
    }
    func writeMappedBytes(id: ResourceID, offset: UInt64, bytes: [UInt8]) async {
        events.append("write")
        await writing.release()
        if let completeWriting { await completeWriting.wait() }
        events.append("write-complete")
    }
    func endBufferMapping(id: ResourceID) { events.append("unmap") }
}

@Test func unmapWaitsForAcceptedMappedWritesBeforeReleasingTheBuffer() async throws {
    let host = MappingTestHost(blockWrites: true)
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    let buffer = try await device.createBuffer(.init(size: 16, usage: [.mapWrite, .copySource]))
    await host.completeOpening.release()
    let mapping = try await device.mapBuffer(buffer, mode: .write)
    let write = Task { try await device.writeMappedBytes(mapping, offset: 0, bytes: [1, 2, 3]) }
    await host.writing.wait()
    let close = Task { try await device.unmapBuffer(buffer) }
    while try await device.bufferMapState(buffer) == .mapped { await Task.yield() }
    #expect(try await device.bufferMapState(buffer) == .pending)
    await #expect(throws: GPUError.self) { try await device.submitResourceUse([buffer]) }
    #expect(await host.events == ["map", "write"])
    await host.completeWriting?.release()
    try await write.value
    try await close.value
    #expect(try await device.bufferMapState(buffer) == .unmapped)
    #expect(await host.events == ["map", "write", "write-complete", "unmap"])
    await device.close()
}

@Test func pendingMapWaitsForGPUUseAndCanBeCancelledBeforeNativeMapping() async throws {
    let host = MappingTestHost()
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    let buffer = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let submission = try await device.submitResourceUse([buffer])
    await host.submitted.wait()
    let pending = Task { try await device.mapBuffer(buffer, mode: .read) }
    while try await device.bufferMapState(buffer) == .unmapped { await Task.yield() }
    #expect(try await device.bufferMapState(buffer) == .pending)
    #expect(await host.events.isEmpty)
    await #expect(throws: GPUError.self) { try await device.submitResourceUse([buffer]) }
    try await device.unmapBuffer(buffer)
    #expect(try await device.bufferMapState(buffer) == .unmapped)
    await host.completeOpening.release()
    await host.completeSubmission.release()
    try await submission.wait()
    await #expect(throws: CancellationError.self) { try await pending.value }
    #expect(await host.events.isEmpty)

    let mapped = try await device.mapBuffer(buffer, mode: .read, offset: 8, size: 4)
    #expect(try await device.readMappedBytes(mapped) == [42, 42, 42, 42])
    await #expect(throws: GPUError.self) {
        try await device.writeMappedBytes(mapped, offset: 8, bytes: [0])
    }
    try await device.unmapBuffer(buffer)
    let remapped = try await device.mapBuffer(buffer, mode: .read, offset: 0, size: 0)
    #expect(try await device.readMappedBytes(remapped).isEmpty)
    await #expect(throws: GPUError.self) { try await device.readMappedBytes(mapped) }
    try await device.destroyBuffer(buffer)
    #expect(await host.events == ["map", "unmap", "map", "unmap", "destroy"])
    await device.close()
}

@Test func destroyingAnOpeningMapRetiresNativeMappingBeforeTheAllocation() async throws {
    let host = MappingTestHost()
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    let buffer = try await device.createBuffer(.init(size: 16, usage: .mapRead))
    let pending = Task { try await device.mapBuffer(buffer, mode: .read) }
    await host.opening.wait()
    try await device.destroyBuffer(buffer)
    #expect(await host.events == ["map"])
    await host.completeOpening.release()
    await #expect(throws: CancellationError.self) { try await pending.value }
    #expect(await host.events == ["map", "unmap", "destroy"])
    await device.close()
}

#if canImport(Metal)
@Test func mappedBytesRoundTripThroughGPUBufferCopiesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let upload = try await device.createBuffer(.init(size: 16, usage: [.mapWrite, .copySource], mappedAtCreation: true))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let initial = try await device.mappedBuffer(upload)
    #expect(try await device.readMappedBytes(initial) == Array(repeating: 0, count: 16))
    try await device.writeMappedBytes(initial, offset: 1, bytes: [1, 2, 3, 4, 5, 6, 7, 8])
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.copy(source: upload, destination: readback, byteCount: 16)])
    }
    try await device.unmapBuffer(upload)
    let second = try await device.mapBuffer(upload, mode: .write, offset: 8, size: 8)
    #expect(try await device.readMappedBytes(second) == [8, 0, 0, 0, 0, 0, 0, 0])
    try await device.writeMappedBytes(second, offset: 9, bytes: [9, 10, 11])
    try await device.unmapBuffer(upload)

    let copy = try await device.submitCommands([.copy(source: upload, destination: readback, byteCount: 16)])
    await #expect(throws: GPUError.self) { try await device.mapBuffer(readback, mode: .read, offset: 4, size: 4) }
    await #expect(throws: GPUError.self) { try await device.mapBuffer(readback, mode: .read, offset: 8, size: 6) }
    await #expect(throws: GPUError.self) { try await device.mapBuffer(readback, mode: .write) }
    let result = try await device.mapBuffer(readback, mode: .read, offset: 8)
    #expect(try await device.readMappedBytes(result) == [8, 9, 10, 11, 0, 0, 0, 0])
    #expect(try await device.readMappedBytes(result, offset: 9, size: 3) == [9, 10, 11])
    await #expect(throws: GPUError.self) { try await device.readMappedBytes(result, offset: 7, size: 1) }
    try await device.unmapBuffer(readback)
    _ = try await copy.wait()

    let initialized = try await device.createBuffer(.init(size: 4, usage: .copySource, mappedAtCreation: true))
    let creation = try await device.mappedBuffer(initialized)
    try await device.writeMappedBytes(creation, offset: 0, bytes: [21, 22, 23, 24])
    try await device.unmapBuffer(initialized)
    _ = try await device.submitCommands([.copy(source: initialized, destination: readback, byteCount: 4)])
    let prefix = try await device.mapBuffer(readback, mode: .read, size: 4)
    #expect(try await device.readMappedBytes(prefix) == [21, 22, 23, 24])
    await device.close()
}
#endif
