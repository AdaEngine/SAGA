import Testing
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGATypes

private actor MeshPipelineStageLimitBackend: MeshBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(
        maxBufferSize: 1024, features: [.meshShaders],
        maxTaskInvocationsPerWorkgroup: 12, maxTaskInvocationsPerDimension: 6,
        maxMeshInvocationsPerWorkgroup: 8, maxMeshInvocationsPerDimension: 4,
        maxTaskPayloadSize: 16, maxMeshOutputVertices: 4, maxMeshOutputPrimitives: 2)
    private(set) var allocations = 0
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws { allocations += 1 }
    func destroyMeshPipeline(id: ResourceID) async {}
}

@Test func meshPipelineValidatesStageLimitsBeforeBackendAllocation() async throws {
    let backend = MeshPipelineStageLimitBackend()
    let device = try await Device.make(backend: backend)
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs", colorOutputKinds: [0: .float])
    func mesh(_ size: IRWorkgroupSize = .init(), payload: MeshPayloadLayout? = nil, vertices: UInt32 = 4, primitives: UInt32 = 2) -> MeshStageArtifact {
        .init(artifact: code, sourceEntryPoint: "mesh", workgroupSize: size, payload: payload, topology: .triangles,
              maxVertices: vertices, maxPrimitives: primitives, vertexVaryings: [], primitiveVaryings: [])
    }
    func task(_ size: IRWorkgroupSize = .init(), payload: MeshPayloadLayout? = nil) -> TaskStageArtifact {
        .init(artifact: code, sourceEntryPoint: "task", workgroupSize: size, payload: payload)
    }
    let payload = MeshPayloadLayout(size: 16, members: [])
    _ = try await device.createMeshPipeline(task: task(.init(x: 6, y: 2), payload: payload), mesh: mesh(.init(x: 4, y: 2), payload: payload), fragment: fragment, colorFormat: .rgba8Unorm)
    for size in [IRWorkgroupSize(x: 5), .init(x: 3, y: 3), .init(x: 0), .init(x: .max, y: .max, z: .max)] {
        await #expect(throws: GPUError.self) { try await device.createMeshPipeline(mesh: mesh(size), fragment: fragment, colorFormat: .rgba8Unorm) }
    }
    for size in [IRWorkgroupSize(x: 7), .init(x: 4, y: 4), .init(x: 0), .init(x: .max, y: .max, z: .max)] {
        await #expect(throws: GPUError.self) { try await device.createMeshPipeline(task: task(size), mesh: mesh(), fragment: fragment, colorFormat: .rgba8Unorm) }
    }
    let oversized = MeshPayloadLayout(size: 17, members: [])
    await #expect(throws: GPUError.self) { try await device.createMeshPipeline(task: task(payload: oversized), mesh: mesh(payload: oversized), fragment: fragment, colorFormat: .rgba8Unorm) }
    await #expect(throws: GPUError.self) { try await device.createMeshPipeline(mesh: mesh(vertices: 5), fragment: fragment, colorFormat: .rgba8Unorm) }
    await #expect(throws: GPUError.self) { try await device.createMeshPipeline(mesh: mesh(primitives: 3), fragment: fragment, colorFormat: .rgba8Unorm) }
    #expect(await backend.allocations == 1)
    await device.close()
}
