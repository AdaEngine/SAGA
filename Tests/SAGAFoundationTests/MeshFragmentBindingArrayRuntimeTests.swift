import Testing
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGATypes

private actor FragmentArrayBackend: MeshBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders])
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyMeshPipeline(id: ResourceID) async {}
}

@Test func meshFragmentArraysExpandRequirementsAndRejectOverlappingBindings() async throws {
    let device = try await Device.make(backend: FragmentArrayBackend())
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [])
    let array = ComputeShaderArtifact.BindingArrayLayout(count: 2, hlslRegisterBase: nil)
    let buffer = ComputeShaderArtifact.StorageBinding(binding: .init(group: 0, binding: 1), flatBindingIndex: 0,
        minimumByteSize: 16, isReadOnly: true, bindingArray: array)
    let texture = ComputeShaderArtifact.SampledTextureBinding(binding: .init(group: 0, binding: 2), flatBindingIndex: 0, bindingArray: array)
    let sampler = ComputeShaderArtifact.SamplerBinding(binding: .init(group: 0, binding: 3), flatBindingIndex: 0, bindingArray: array)
    let storage = ComputeShaderArtifact.StorageTextureBinding(binding: .init(group: 0, binding: 4), format: .rgba8Unorm,
        flatBindingIndex: 2, bindingArray: array)
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs",
        storageBindings: [buffer], sampledTextures: [texture], storageTextures: [storage], samplerBindings: [sampler], colorOutputKinds: [0: .float])
    let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    #expect(pipeline.fragmentBufferBindings.map(\.slot) == (0..<2).map { .init(binding: 1, arrayElement: UInt32($0)) })
    #expect(pipeline.fragmentTextures.map(\.slot) == (0..<2).map { .init(binding: 2, arrayElement: UInt32($0)) })
    #expect(pipeline.fragmentSamplers.map(\.slot) == (0..<2).map { .init(binding: 3, arrayElement: UInt32($0)) })
    #expect(pipeline.fragmentStorageTextures.map(\.slot) == (0..<2).map { .init(binding: 4, arrayElement: UInt32($0)) })
    let overlap = ComputeShaderArtifact.SamplerBinding(binding: .init(group: 0, binding: 2), flatBindingIndex: 0, bindingArray: array)
    let invalid = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs",
        sampledTextures: [texture], samplerBindings: [overlap], colorOutputKinds: [0: .float])
    await #expect(throws: GPUError.self) { try await device.createMeshPipeline(mesh: mesh, fragment: invalid, colorFormat: .rgba8Unorm) }
    await device.close()
}
