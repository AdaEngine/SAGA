import Testing
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGATypes

private actor NumericFeatureBackend: MeshBackendDevice {
    nonisolated let capabilities: DeviceCapabilities
    private(set) var allocations = 0
    init(features: Set<GPUFeature>) throws {
        capabilities = try DeviceCapabilities(maxBufferSize: 1024, features: features)
    }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws { allocations += 1 }
    func destroyMeshPipeline(id: ResourceID) async {}
}

@Test func taskAndMeshNumericFeaturesAreRequiredBeforeAllocation() async throws {
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs", colorOutputKinds: [0: .float])
    let features: [GPUFeature] = [.shaderF16, .shaderInt16, .shaderInt64, .shaderInt64AtomicMinMax, .shaderInt64AtomicAllOps]
    for isTask in [false, true] {
        for feature in features {
            let task = isTask ? TaskStageArtifact(artifact: code, sourceEntryPoint: "task", workgroupSize: .init(), payload: nil,
                requiresShaderF16: feature == .shaderF16, requiresShaderInt16: feature == .shaderInt16,
                requiresShaderInt64: feature == .shaderInt64, requiresShaderInt64AtomicMinMax: feature == .shaderInt64AtomicMinMax,
                requiresShaderInt64AtomicAllOps: feature == .shaderInt64AtomicAllOps) : nil
            let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
                topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [],
                requiresShaderF16: !isTask && feature == .shaderF16, requiresShaderInt16: !isTask && feature == .shaderInt16,
                requiresShaderInt64: !isTask && feature == .shaderInt64, requiresShaderInt64AtomicMinMax: !isTask && feature == .shaderInt64AtomicMinMax,
                requiresShaderInt64AtomicAllOps: !isTask && feature == .shaderInt64AtomicAllOps)
            let rejectedBackend = try NumericFeatureBackend(features: [.meshShaders])
            let rejected = try await Device.make(backend: rejectedBackend)
            await #expect(throws: GPUError.self) {
                try await rejected.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
            }
            #expect(await rejectedBackend.allocations == 0)
            await rejected.close()
            let acceptedBackend = try NumericFeatureBackend(features: [.meshShaders, feature])
            let accepted = try await Device.make(backend: acceptedBackend)
            _ = try await accepted.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
            #expect(await acceptedBackend.allocations == 1)
            await accepted.close()
        }
    }
}

@Test func taskMeshInt64AllOpsSatisfiesMinMax() async throws {
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [], requiresShaderInt64AtomicMinMax: true)
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs", colorOutputKinds: [0: .float])
    let backend = try NumericFeatureBackend(features: [.meshShaders, .shaderInt64AtomicAllOps])
    let device = try await Device.make(backend: backend)
    _ = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    #expect(await backend.allocations == 1)
    await device.close()
}
