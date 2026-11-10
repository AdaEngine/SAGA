import Testing
import SAGATypes

@Test func meshDrawRejectsInvalidAttachmentAndForeignPipeline() throws {
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm)
    let target = ResourceID(device: 1, serial: 2)
    let draw = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1))
    let good = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment])
    try draw.validate(device: 1, descriptors: [:], textures: [target: good])
    #expect(throws: GPUError.self) { try draw.validate(device: 2, descriptors: [:], textures: [target: good]) }
    let bad = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.textureBinding])
    #expect(throws: GPUError.self) { try draw.validate(device: 1, descriptors: [:], textures: [target: bad]) }
}

import SAGABackend
import SAGACore
import SAGAShaderIR

private actor MeshLifetimeBackend: MeshBackendDevice, RenderBackendDevice, SamplerBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders])
    private var continuation: CheckedContinuation<Void, Never>?
    private var started = false
    private(set) var destroyedPipelines: [ResourceID] = []
    private(set) var destroyedTextures: [ResourceID] = []
    private(set) var destroyedBuffers: [ResourceID] = []
    private(set) var destroyedSamplers: [ResourceID] = []
    private(set) var submitted = 0
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async { destroyedBuffers.append(id) }
    func submit(_ submission: Submission) async throws {}
    func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws {}
    func destroyTexture(id: ResourceID) async { destroyedTextures.append(id) }
    func allocateSampler(id: ResourceID, descriptor: SamplerDescriptor) async throws {}
    func destroySampler(id: ResourceID) async { destroyedSamplers.append(id) }
    func allocateRenderPipeline(_ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyRenderPipeline(id: ResourceID) async {}
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyMeshPipeline(id: ResourceID) async { destroyedPipelines.append(id) }
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        submitted += 1
        started = true
        await withCheckedContinuation { continuation = $0 }
        return []
    }
    func waitForStart() async { while !started { await Task.yield() } }
    func release() { continuation?.resume(); continuation = nil }
}

private actor NoMeshBackend: BackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders])
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
}

private func meshTestArtifacts() -> (MeshStageArtifact, RenderStageArtifact) {
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
                                 topology: .triangles, maxVertices: 3, maxPrimitives: 1,
                                 vertexVaryings: [], primitiveVaryings: [])
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs",
                                       colorOutputKinds: [0: .float])
    return (mesh, fragment)
}

@Test func meshDrawPinsResourcesUntilAcceptedWorkRetires() async throws {
    let backend = MeshLifetimeBackend()
    let device = try await Device.make(backend: backend)
    let (mesh, fragment) = meshTestArtifacts()
    let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormats: [0: .rgba8Unorm, 1: .rgba8Unorm], depthState: .init())
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: [.renderAttachment]))
    let second = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: [.renderAttachment]))
    let depth = try await device.createTexture(.init(width: 2, height: 2, format: .depth32Float, usage: [.renderAttachment]))
    let wrong = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint, usage: [.renderAttachment]))
    let command = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), additionalColors: [1: .init(texture: second)], depth: .init(texture: depth))
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.drawMesh(pipeline: pipeline, target: wrong, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1))])
    }
    let foreign = MeshPipeline(id: .init(device: pipeline.id.device + 1, serial: pipeline.id.serial), colorFormat: .rgba8Unorm)
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.drawMesh(pipeline: foreign, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1))])
    }
    #expect(await backend.submitted == 0)
    let ticket = try await device.submitCommands([command])
    await backend.waitForStart()
    let waiter = Task { try await ticket.wait() }
    waiter.cancel()
    try await device.destroyTexture(target)
    try await device.destroyTexture(second)
    try await device.destroyTexture(depth)
    try await device.destroyMeshPipeline(pipeline.id)
    #expect(!(await backend.destroyedTextures).contains(target))
    #expect(!(await backend.destroyedTextures).contains(second))
    #expect(!(await backend.destroyedTextures).contains(depth))
    #expect(!(await backend.destroyedPipelines).contains(pipeline.id))
    await backend.release()
    _ = try await ticket.wait()
    await #expect(throws: CancellationError.self) { try await waiter.value }
    #expect((await backend.destroyedTextures).contains(target))
    #expect((await backend.destroyedTextures).contains(second))
    #expect((await backend.destroyedTextures).contains(depth))
    #expect((await backend.destroyedPipelines).contains(pipeline.id))
    await device.close()
}

@Test func meshPipelineRejectsBackendWithoutMeshSupport() async throws {
    let device = try await Device.make(backend: NoMeshBackend())
    let (mesh, fragment) = meshTestArtifacts()
    await #expect(throws: GPUError.self) {
        try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    }
    await device.close()
}

@Test func meshDrawPinsBoundBufferUntilRetirement() async throws {
    let backend = MeshLifetimeBackend()
    let device = try await Device.make(backend: backend)
    let (original, fragment) = meshTestArtifacts()
    let binding = ComputeShaderArtifact.StorageBinding(binding: .init(group: 0, binding: 2), flatBindingIndex: 0,
                                                        minimumByteSize: 16, isReadOnly: false)
    let mesh = MeshStageArtifact(artifact: original.artifact, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
                                 topology: .triangles, maxVertices: 3, maxPrimitives: 1,
                                 vertexVaryings: [], primitiveVaryings: [], storageBindings: [binding])
    let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: [.renderAttachment]))
    let buffer = try await device.createBuffer(.init(size: 256, usage: .storage))
    let ticket = try await device.submitCommands([.drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack),
        workgroups: .init(x: 1), meshBuffers: [.init(binding: 2): .init(buffer: buffer, size: 16)])])
    await backend.waitForStart()
    try await device.destroyBuffer(buffer)
    #expect(!(await backend.destroyedBuffers).contains(buffer))
    await backend.release()
    _ = try await ticket.wait()
    #expect((await backend.destroyedBuffers).contains(buffer))
    await device.close()
}

@Test func meshDrawPinsSampledTextureAndSamplerUntilRetirement() async throws {
    let backend = MeshLifetimeBackend()
    let device = try await Device.make(backend: backend)
    let (original, fragment) = meshTestArtifacts()
    let textureBinding = ComputeShaderArtifact.SampledTextureBinding(binding: .init(group: 0, binding: 2), flatBindingIndex: 0)
    let samplerBinding = ComputeShaderArtifact.SamplerBinding(binding: .init(group: 0, binding: 3), flatBindingIndex: 0)
    let mesh = MeshStageArtifact(artifact: original.artifact, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [],
        sampledTextures: [textureBinding], samplerBindings: [samplerBinding])
    let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: [.renderAttachment]))
    let sampled = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: [.textureBinding]))
    let sampler = try await device.createSampler()
    let ticket = try await device.submitCommands([.drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack),
        workgroups: .init(x: 1), meshTextures: [.init(binding: 2): sampled], meshSamplers: [.init(binding: 3): sampler])])
    await backend.waitForStart()
    try await device.destroyTexture(sampled)
    try await device.destroySampler(sampler)
    #expect(!(await backend.destroyedTextures).contains(sampled))
    #expect(!(await backend.destroyedSamplers).contains(sampler))
    await backend.release()
    _ = try await ticket.wait()
    #expect((await backend.destroyedTextures).contains(sampled))
    #expect((await backend.destroyedSamplers).contains(sampler))
    await device.close()
}

@Test func meshMultipleTargetsValidateAndPin() throws {
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormats: [0: .rgba8Unorm, 1: .rgba8Uint])
    let first = ResourceID(device: 1, serial: 2)
    let second = ResourceID(device: 1, serial: 3)
    let draw = BufferCommand.drawMesh(pipeline: pipeline, target: first, colorLoad: .load, workgroups: .init(x: 1),
                                      additionalColors: [1: .init(texture: second, loadAction: .clear(.transparentBlack))])
    let textures = [first: TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment]),
                    second: TextureDescriptor(width: 4, height: 4, format: .rgba8Uint, usage: [.renderAttachment])]
    try draw.validate(device: 1, descriptors: [:], textures: textures)
    #expect(draw.resources.contains(second))
    let missing = BufferCommand.drawMesh(pipeline: pipeline, target: first, colorLoad: .load, workgroups: .init(x: 1))
    #expect(throws: GPUError.self) { try missing.validate(device: 1, descriptors: [:], textures: textures) }
    let aliased = BufferCommand.drawMesh(pipeline: pipeline, target: first, colorLoad: .load, workgroups: .init(x: 1), additionalColors: [1: .init(texture: first)])
    #expect(throws: GPUError.self) { try aliased.validate(device: 1, descriptors: [:], textures: textures) }
}

@Test func meshIndirectPinsArgumentBufferUntilRetirement() async throws {
    let backend = MeshLifetimeBackend()
    let device = try await Device.make(backend: backend)
    let (mesh, fragment) = meshTestArtifacts()
    let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: [.renderAttachment]))
    let arguments = try await device.createBuffer(.init(size: 12, usage: [.indirect]))
    let ticket = try await device.submitCommands([.drawMeshIndirect(pipeline: pipeline, target: target, colorLoad: .load, indirectBuffer: arguments)])
    await backend.waitForStart()
    try await device.destroyBuffer(arguments)
    #expect(!(await backend.destroyedBuffers).contains(arguments))
    await backend.release()
    _ = try await ticket.wait()
    #expect((await backend.destroyedBuffers).contains(arguments))
    await device.close()
}
