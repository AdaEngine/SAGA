import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true), arguments: [false, true])
func metalRejectsExcessiveMeshStageThreadgroupStorage(taskStage: Bool) async throws {
    let native = try #require(MTLCreateSystemDefaultDevice())
    let excessive = UInt64(native.maxThreadgroupMemoryLength) + 1
    let code = ShaderArtifact.text(format: .msl, source: "invalid Metal source")
    let task = TaskStageArtifact(artifact: code, sourceEntryPoint: "task", workgroupSize: .init(),
                                 workgroupStorageByteSize: taskStage ? excessive : 0, payload: nil)
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(),
                                 workgroupStorageByteSize: taskStage ? 0 : excessive, payload: nil,
                                 topology: .triangles, maxVertices: 3, maxPrimitives: 1,
                                 vertexVaryings: [], primitiveVaryings: [])
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs",
                                       colorOutputKinds: [0: .float])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    await #expect(throws: GPUError.validation("Metal mesh or object threadgroup storage exceeds the device limit")) {
        try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    }
    await device.close()
}
#endif
