import Testing
import SAGATypes
import SAGABackend
import SAGACore

private actor AccelerationResourceBackend: AccelerationStructureBackendDevice, CommandBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    private(set) var allocated: [ResourceID] = []
    private(set) var destroyed: [ResourceID] = []
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateAccelerationStructure(id: ResourceID, descriptor: AccelerationStructureDescriptor) async throws { allocated.append(id) }
    func destroyAccelerationStructure(id: ResourceID) async { destroyed.append(id) }
}

@Test func accelerationStructureResourceValidatesAndRetires() async throws {
    let backend = AccelerationResourceBackend()
    let device = try await Device.make(backend: backend)
    await #expect(throws: GPUError.self) { try await device.createAccelerationStructure(.init(maxTriangleCount: 0)) }
    let id = try await device.createAccelerationStructure(.init(maxTriangleCount: 2))
    try await device.validateResource(id)
    await #expect(throws: GPUError.self) { try await device.destroyBuffer(id) }
    try await device.destroyAccelerationStructure(id)
    try await device.destroyAccelerationStructure(id)
    #expect(await backend.allocated == [id])
    #expect(await backend.destroyed == [id])
    await #expect(throws: GPUError.self) { try await device.validateResource(id) }
    await device.close()
}

@Test func accelerationBuildRecordsBothResourcesAndChecksGeometry() throws {
    let buffer = ResourceID(device: 1, serial: 1)
    let destination = ResourceID(device: 1, serial: 2)
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: buffer, vertexStride: 12, vertexCount: 3)
    let command = BufferCommand.buildAccelerationStructure(destination: destination, geometry: geometry)
    #expect(command.resources == [destination, buffer])
    try command.validate(device: 1, descriptors: [buffer: .init(size: 36, usage: .vertex)])
    #expect(throws: GPUError.self) {
        try command.validate(device: 1, descriptors: [buffer: .init(size: 35, usage: .vertex)])
    }
    #expect(throws: GPUError.self) {
        try command.validate(device: 1, descriptors: [buffer: .init(size: 36, usage: .storage)])
    }
}

@Test func accelerationBuildRejectsExcessCapacityAndForeignDestination() async throws {
    let backend = AccelerationResourceBackend()
    let device = try await Device.make(backend: backend)
    let other = try await Device.make(backend: backend)
    let buffer = try await device.createBuffer(.init(size: 72, usage: .vertex))
    let structure = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let foreign = try await other.createAccelerationStructure(.init(maxTriangleCount: 2))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: buffer, vertexStride: 12, vertexCount: 6)
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.buildAccelerationStructure(destination: structure, geometry: geometry)])
    }
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.buildAccelerationStructure(destination: foreign, geometry: geometry)])
    }
    let valid = AccelerationStructureTriangleGeometry(vertexBuffer: buffer, vertexStride: 12, vertexCount: 3)
    let ticket = try await device.submitCommands([.buildAccelerationStructure(destination: structure, geometry: valid)])
    _ = try await ticket.wait()
    await device.close()
    await other.close()
}

@Test func topLevelBuildValidatesInstancesAndResourceKinds() async throws {
    let backend = AccelerationResourceBackend()
    let device = try await Device.make(backend: backend)
    let blas = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let tlas = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let instance = AccelerationStructureInstance(accelerationStructure: blas)
    let command = BufferCommand.buildTopLevelAccelerationStructure(destination: tlas, instances: [instance])
    #expect(command.resources == [tlas, blas])
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.buildTopLevelAccelerationStructure(destination: blas, instances: [instance])])
    }
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.buildTopLevelAccelerationStructure(destination: tlas, instances: [instance, instance])])
    }
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.buildTopLevelAccelerationStructure(destination: tlas, instances: [.init(accelerationStructure: tlas)])])
    }
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.buildTopLevelAccelerationStructure(destination: tlas, instances: [.init(accelerationStructure: blas, transform: [1])])])
    }
    for invalid in [
        AccelerationStructureInstance(accelerationStructure: blas, transform: []),
        AccelerationStructureInstance(accelerationStructure: blas, transform: [.nan] + Array(repeating: 0, count: 11)),
        AccelerationStructureInstance(accelerationStructure: blas, instanceID: 0x0100_0000),
    ] {
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.buildTopLevelAccelerationStructure(destination: tlas, instances: [invalid])])
        }
    }
    let vertices = try await device.createBuffer(.init(size: 36, usage: .vertex))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertices, vertexStride: 12, vertexCount: 3)
    let bottomBuild = try await device.submitCommands([.buildAccelerationStructure(destination: blas, geometry: geometry)])
    _ = try await bottomBuild.wait()
    let boundary = AccelerationStructureInstance(accelerationStructure: blas, mask: .max, instanceID: 0x00ff_ffff)
    let boundaryTicket = try await device.submitCommands([.buildTopLevelAccelerationStructure(destination: tlas, instances: [boundary])])
    _ = try await boundaryTicket.wait()
    let ticket = try await device.submitCommands([command])
    _ = try await ticket.wait()
    try await device.destroyAccelerationStructure(blas)
    #expect(!(await backend.destroyed).contains(blas))
    try await device.destroyAccelerationStructure(tlas)
    #expect((await backend.destroyed).contains(blas))
    await device.close()
}

@Test func topLevelBuildRejectsUnbuiltBottomLevelStructure() async throws {
    let backend = AccelerationResourceBackend()
    let device = try await Device.make(backend: backend)
    let bottom = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    await #expect(throws: GPUError.validation("Top-level acceleration structure references an unbuilt BLAS")) {
        try await device.submitCommands([.buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: bottom)])])
    }
    await device.close()
}

@Test func topLevelRebuildReleasesPreviousBottomLevelStructure() async throws {
    let backend = AccelerationResourceBackend()
    let device = try await Device.make(backend: backend)
    let first = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let second = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let vertices = try await device.createBuffer(.init(size: 36, usage: .vertex))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertices, vertexStride: 12, vertexCount: 3)
    let bottomBuild = try await device.submitCommands([
        .buildAccelerationStructure(destination: first, geometry: geometry),
        .buildAccelerationStructure(destination: second, geometry: geometry)
    ])
    _ = try await bottomBuild.wait()
    for source in [first, second] {
        let ticket = try await device.submitCommands([
            .buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: source)])
        ])
        _ = try await ticket.wait()
    }
    try await device.destroyAccelerationStructure(first)
    try await device.destroyAccelerationStructure(second)
    #expect((await backend.destroyed).contains(first))
    #expect(!(await backend.destroyed).contains(second))
    try await device.destroyAccelerationStructure(top)
    #expect(Set(await backend.destroyed) == Set([top, first, second]))
    await device.close()
}

@Test func computeAccelerationStructureBindingsValidateAndPin() throws {
    let slot = ComputeBindingSlot(group: 1, binding: 2)
    let top = ResourceID(device: 1, serial: 2)
    let pipeline = ComputePipeline(id: .init(device: 1, serial: 1), storageBindings: [],
                                   accelerationStructureBindings: [slot])
    let command = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                                                accelerationStructures: [slot: top])
    #expect(command.resources.contains(top))
    try command.validate(device: 1, descriptors: [:], accelerationStructures: [top: .init(maxInstanceCount: 1)])
    #expect(throws: GPUError.self) {
        try command.validate(device: 1, descriptors: [:], accelerationStructures: [top: .init(maxTriangleCount: 1)])
    }
    #expect(throws: GPUError.self) {
        try command.validate(device: 1, descriptors: [:])
    }
    let wrongSlot = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                                                  accelerationStructures: [.init(binding: 9): top])
    #expect(throws: GPUError.self) {
        try wrongSlot.validate(device: 1, descriptors: [:], accelerationStructures: [top: .init(maxInstanceCount: 1)])
    }
    let missing = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1))
    #expect(throws: GPUError.self) {
        try missing.validate(device: 1, descriptors: [:], accelerationStructures: [top: .init(maxInstanceCount: 1)])
    }
}

@Test func computeAccelerationStructureRequiresBuildAndAcceptsOrderedBuild() async throws {
    let backend = AccelerationResourceBackend()
    let device = try await Device.make(backend: backend)
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let slot = ComputeBindingSlot(binding: 2)
    let pipeline = ComputePipeline(id: .init(device: top.device, serial: 99), storageBindings: [],
                                   accelerationStructureBindings: [slot])
    let dispatch = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                                                 accelerationStructures: [slot: top])
    await #expect(throws: GPUError.self) { try await device.submitCommands([dispatch]) }
    let build = BufferCommand.buildTopLevelAccelerationStructure(destination: top, instances: [])
    let first = try await device.submitCommands([build, dispatch])
    _ = try await first.wait()
    let second = try await device.submitCommands([dispatch])
    _ = try await second.wait()
    await device.close()
}

private actor GatedAccelerationBackend: AccelerationStructureBackendDevice, CommandBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    private var waiter: CheckedContinuation<Void, Never>?
    private var started = false
    private var released = false
    private(set) var calls = 0
    private(set) var destroyed: [ResourceID] = []
    let failBuild: Bool
    init(failBuild: Bool) { self.failBuild = failBuild }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func allocateAccelerationStructure(id: ResourceID, descriptor: AccelerationStructureDescriptor) async throws {}
    func destroyAccelerationStructure(id: ResourceID) async { destroyed.append(id) }
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        calls += 1
        if calls == 1 {
            started = true
            if !released { await withCheckedContinuation { waiter = $0 } }
            if failBuild { throw GPUError.validation("build failed") }
        }
        return []
    }
    func waitForStart() async { while !started { await Task.yield() } }
    func release() { released = true; waiter?.resume(); waiter = nil }
}

@Test func failedAccelerationBuildBlocksQueuedDispatch() async throws {
    let backend = GatedAccelerationBackend(failBuild: true)
    let device = try await Device.make(backend: backend)
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let slot = ComputeBindingSlot(binding: 0)
    let pipeline = ComputePipeline(id: .init(device: top.device, serial: 99), storageBindings: [],
                                   accelerationStructureBindings: [slot])
    let build = try await device.submitCommands([.buildTopLevelAccelerationStructure(destination: top, instances: [])])
    await backend.waitForStart()
    let dispatch = try await device.submitCommands([.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                                                                     accelerationStructures: [slot: top])])
    await backend.release()
    await #expect(throws: GPUError.self) { try await build.wait() }
    await #expect(throws: GPUError.self) { try await dispatch.wait() }
    #expect(await backend.calls == 1)
    await device.close()
}

@Test func failedBottomLevelBuildBlocksQueuedTopLevelBuild() async throws {
    let backend = GatedAccelerationBackend(failBuild: true)
    let device = try await Device.make(backend: backend)
    let vertices = try await device.createBuffer(.init(size: 36, usage: .vertex))
    let bottom = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertices, vertexStride: 12, vertexCount: 3)
    let first = try await device.submitCommands([.buildAccelerationStructure(destination: bottom, geometry: geometry)])
    await backend.waitForStart()
    let second = try await device.submitCommands([
        .buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: bottom)])
    ])
    await backend.release()
    await #expect(throws: GPUError.self) { try await first.wait() }
    await #expect(throws: GPUError.self) { try await second.wait() }
    #expect(await backend.calls == 1)
    await device.close()
}

@Test func accelerationDispatchRejectsForeignDestroyedAndPinsAcceptedResource() async throws {
    let backend = GatedAccelerationBackend(failBuild: false)
    let device = try await Device.make(backend: backend)
    let other = try await Device.make(backend: AccelerationResourceBackend())
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let foreign = try await other.createAccelerationStructure(.init(maxInstanceCount: 1))
    let slot = ComputeBindingSlot(binding: 0)
    let pipeline = ComputePipeline(id: .init(device: top.device, serial: 99), storageBindings: [],
                                   accelerationStructureBindings: [slot])
    func dispatch(_ id: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1), accelerationStructures: [slot: id])
    }
    await #expect(throws: GPUError.self) { try await device.submitCommands([dispatch(foreign)]) }
    let build = try await device.submitCommands([.buildTopLevelAccelerationStructure(destination: top, instances: [])])
    await backend.waitForStart()
    let ticket = try await device.submitCommands([dispatch(top)])
    try await device.destroyAccelerationStructure(top)
    #expect(await backend.destroyed.isEmpty)
    await #expect(throws: GPUError.self) { try await device.submitCommands([dispatch(top)]) }
    await backend.release()
    _ = try await build.wait()
    _ = try await ticket.wait()
    #expect(await backend.destroyed == [top])
    await device.close()
    await other.close()
}

@Test func indexedAccelerationBuildValidatesUsageBoundsAndCapacity() async throws {
    let backend = AccelerationResourceBackend()
    let device = try await Device.make(backend: backend)
    let vertex = try await device.createBuffer(.init(size: 48, usage: .vertex))
    let index = try await device.createBuffer(.init(size: 12, usage: .index))
    let wrong = try await device.createBuffer(.init(size: 12, usage: .storage))
    let bottom = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    func geometry(_ id: ResourceID, _ count: UInt32) -> AccelerationStructureTriangleGeometry {
        .init(vertexBuffer: vertex, vertexStride: 12, vertexCount: 4,
              indexBuffer: .init(buffer: id, format: .uint16), indexCount: count)
    }
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.buildAccelerationStructure(destination: bottom, geometry: geometry(wrong, 3))])
    }
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.buildAccelerationStructure(destination: bottom, geometry: geometry(index, 6))])
    }
    let ticket = try await device.submitCommands([.buildAccelerationStructure(destination: bottom, geometry: geometry(index, 3))])
    _ = try await ticket.wait()
    await device.close()
}
