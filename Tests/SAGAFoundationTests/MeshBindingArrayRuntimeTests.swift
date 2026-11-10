import Testing
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGATypes

private actor MeshArrayLifetimeBackend: MeshBackendDevice, RenderBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders])
    private var continuation: CheckedContinuation<Void, Never>?
    private var started = false
    private(set) var destroyedBuffers: [ResourceID] = []
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async { destroyedBuffers.append(id) }
    func submit(_ submission: Submission) async throws {}
    func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws {}
    func destroyTexture(id: ResourceID) async {}
    func allocateRenderPipeline(_ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyRenderPipeline(id: ResourceID) async {}
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyMeshPipeline(id: ResourceID) async {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        started = true
        await withCheckedContinuation { continuation = $0 }
        return []
    }
    func waitForStart() async { while !started { await Task.yield() } }
    func release() { continuation?.resume(); continuation = nil }
}

@Test func meshBindingArrayValidatesEveryElement() throws {
    let first = ComputeBindingSlot(binding: 2, arrayElement: 0)
    let second = ComputeBindingSlot(binding: 2, arrayElement: 1)
    let extra = ComputeBindingSlot(binding: 2, arrayElement: 2)
    let requirements = [first, second].map { ComputePipeline.BufferBinding(slot: $0, minimumByteSize: 16, isReadOnly: true) }
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm, meshBufferBindings: requirements)
    let target = ResourceID(device: 1, serial: 2), a = ResourceID(device: 1, serial: 3)
    let b = ResourceID(device: 1, serial: 4), foreign = ResourceID(device: 2, serial: 4)
    let descriptor = BufferDescriptor(size: 256, usage: .storage)
    let descriptors = [a: descriptor, b: descriptor, foreign: descriptor]
    let textures = [target: TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment)]
    func validate(_ bindings: [ComputeBindingSlot: StorageBufferBinding]) throws {
        try BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1),
                                   meshBuffers: bindings).validate(device: 1, descriptors: descriptors, textures: textures)
    }
    let good: [ComputeBindingSlot: StorageBufferBinding] = [first: .init(buffer: a, size: 16), second: .init(buffer: b, size: 16)]
    try validate(good)
    #expect(throws: GPUError.self) { try validate([first: .init(buffer: a, size: 16)]) }
    #expect(throws: GPUError.self) { try validate(good.merging([extra: .init(buffer: b, size: 16)]) { old, _ in old }) }
    #expect(throws: GPUError.self) { try validate([first: .init(buffer: a, size: 16), second: .init(buffer: foreign, size: 16)]) }
}

@Test func meshBindingArrayElementsRemainPinnedUntilRetirement() async throws {
    let backend = MeshArrayLifetimeBackend()
    let device = try await Device.make(backend: backend)
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let binding = ComputeShaderArtifact.StorageBinding(binding: .init(group: 0, binding: 2), flatBindingIndex: 0,
        minimumByteSize: 16, isReadOnly: true, bindingArray: .init(count: 2, hlslRegisterBase: nil))
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [], storageBindings: [binding])
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs", colorOutputKinds: [0: .float])
    let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let a = try await device.createBuffer(.init(size: 256, usage: .storage))
    let b = try await device.createBuffer(.init(size: 256, usage: .storage))
    let ticket = try await device.submitCommands([.drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1),
        meshBuffers: [.init(binding: 2, arrayElement: 0): .init(buffer: a, size: 16),
                      .init(binding: 2, arrayElement: 1): .init(buffer: b, size: 16)])])
    await backend.waitForStart()
    try await device.destroyBuffer(a)
    try await device.destroyBuffer(b)
    #expect(!(await backend.destroyedBuffers).contains(a))
    #expect(!(await backend.destroyedBuffers).contains(b))
    await backend.release()
    _ = try await ticket.wait()
    #expect((await backend.destroyedBuffers).contains(a))
    #expect((await backend.destroyedBuffers).contains(b))
    await device.close()
}
