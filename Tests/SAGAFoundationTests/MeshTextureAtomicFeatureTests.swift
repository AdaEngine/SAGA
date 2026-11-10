import Testing
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGATypes

private actor MeshAtomicFeatureBackend: MeshBackendDevice {
    nonisolated let capabilities: DeviceCapabilities
    init(atomic: Bool) throws {
        capabilities = try DeviceCapabilities(maxBufferSize: 1024, features: atomic ? [.meshShaders, .textureAtomic] : [.meshShaders])
    }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyMeshPipeline(id: ResourceID) async {}
}

@Test func meshPipelineRequiresTextureAtomicFeatureForEveryStage() async throws {
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let binding = ComputeShaderArtifact.StorageTextureBinding(binding: .init(group: 0, binding: 0), format: .r32Uint,
        access: [.load, .store, .atomic], flatBindingIndex: 0)
    for stage in 0..<3 {
        let task = stage == 0 ? TaskStageArtifact(artifact: code, sourceEntryPoint: "task", workgroupSize: .init(), payload: nil, storageTextures: [binding]) : nil
        let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
            topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [],
            storageTextures: stage == 1 ? [binding] : [])
        let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs",
            storageTextures: stage == 2 ? [binding] : [], colorOutputKinds: [0: .float])
        let unsupported = try await Device.make(backend: MeshAtomicFeatureBackend(atomic: false))
        await #expect(throws: GPUError.self) {
            try await unsupported.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        }
        await unsupported.close()
        let supported = try await Device.make(backend: MeshAtomicFeatureBackend(atomic: true))
        _ = try await supported.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        await supported.close()
    }
}
