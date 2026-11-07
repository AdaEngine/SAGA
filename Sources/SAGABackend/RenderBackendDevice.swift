import SAGATypes
import SAGAShaderIR

public protocol RenderBackendDevice: CommandBackendDevice {
    func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws
    func destroyTexture(id: ResourceID) async
    func allocateRenderPipeline(_ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact) async throws
    func destroyRenderPipeline(id: ResourceID) async
}
