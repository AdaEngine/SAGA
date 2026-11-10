import Testing
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGATypes

private actor MeshFragmentInt64Backend: MeshBackendDevice {
    nonisolated let capabilities: DeviceCapabilities
    init(features: Set<GPUFeature>) throws {
        capabilities = try DeviceCapabilities(maxBufferSize: 1024, features: features)
    }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyMeshPipeline(id: ResourceID) async {}
}

@Test func meshFragmentInt64TextureAtomicsRequireEveryFeature() async throws {
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [])
    let binding = ComputeShaderArtifact.StorageTextureBinding(binding: .init(group: 0, binding: 0), format: .r64Uint,
        access: [.load, .store, .atomic], flatBindingIndex: 0)
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs",
        storageTextures: [binding], colorOutputKinds: [0: .float],
        requiresShaderInt64AtomicMinMax: true, requiresShaderInt64: true)
    let required: Set<GPUFeature> = [.meshShaders, .textureInt64Atomic, .shaderInt64, .shaderInt64AtomicMinMax]
    let missingFeatures: [GPUFeature] = [.textureInt64Atomic, .shaderInt64, .shaderInt64AtomicMinMax]
    for missing in missingFeatures {
        let device = try await Device.make(backend: MeshFragmentInt64Backend(features: required.subtracting([missing])))
        await #expect(throws: GPUError.self) {
            try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        }
        await device.close()
    }
    let device = try await Device.make(backend: MeshFragmentInt64Backend(features: required))
    _ = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    await device.close()
}
