import SAGATypes
import SAGAShaderIR

/// Backend-owned native pipelines live until device teardown or allocation rollback.
public protocol ComputeBackendDevice: CommandBackendDevice {
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) async throws
    func destroyComputePipeline(id: ResourceID) async
}
