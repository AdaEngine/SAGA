import Testing
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGATypes

private actor MeshDispatchLimitBackend: MeshBackendDevice, RenderBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(
        maxBufferSize: 1024, features: [.meshShaders],
        maxTaskWorkgroupTotalCount: 24, maxTaskWorkgroupsPerDimension: 8,
        maxMeshWorkgroupTotalCount: 6, maxMeshWorkgroupsPerDimension: 3)
    private(set) var submissionCount = 0
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws {}
    func destroyTexture(id: ResourceID) async {}
    func allocateRenderPipeline(_ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyRenderPipeline(id: ResourceID) async {}
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyMeshPipeline(id: ResourceID) async {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        submissionCount += 1
        return []
    }
}

@Test func meshDispatchLimitsFlowThroughDeviceSubmission() async throws {
    let backend = MeshDispatchLimitBackend()
    let device = try await Device.make(backend: backend)
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
                                 topology: .triangles, maxVertices: 3, maxPrimitives: 1,
                                 vertexVaryings: [], primitiveVaryings: [])
    let task = TaskStageArtifact(artifact: code, sourceEntryPoint: "task", workgroupSize: .init(), payload: nil)
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs",
                                       colorOutputKinds: [0: .float])
    let meshPipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    let taskPipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: [.renderAttachment]))
    func draw(_ pipeline: MeshPipeline, _ groups: DispatchWorkgroups) -> BufferCommand {
        .drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: groups)
    }
    for (pipeline, groups) in [(meshPipeline, DispatchWorkgroups(x: 3, y: 2)),
                               (taskPipeline, DispatchWorkgroups(x: 4, y: 3, z: 2))] {
        let ticket = try await device.submitCommands([draw(pipeline, groups)])
        _ = try await ticket.wait()
    }
    for (pipeline, groups) in [(meshPipeline, DispatchWorkgroups(x: 4)),
                               (meshPipeline, DispatchWorkgroups(x: 3, y: 3)),
                               (taskPipeline, DispatchWorkgroups(x: 9)),
                               (taskPipeline, DispatchWorkgroups(x: 5, y: 5))] {
        await #expect(throws: GPUError.self) { try await device.submitCommands([draw(pipeline, groups)]) }
    }
    #expect(await backend.submissionCount == 2)
    await device.close()
}
