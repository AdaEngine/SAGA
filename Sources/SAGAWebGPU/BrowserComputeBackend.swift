import SAGATypes
import SAGABackend
import SAGAShaderIR

/// Browser implementation of the compute and buffer SPI, injected by the JS host boundary.
public protocol BrowserGPUComputeHost: BrowserGPUHost, ComputeBackendDevice {}

public protocol BrowserGPURenderHost: BrowserGPUComputeHost, RenderBackendDevice {}

public struct BrowserComputeBackend: ComputeBackendDevice, RenderBackendDevice {
    private let host: any BrowserGPUComputeHost
    public var capabilities: DeviceCapabilities { host.capabilities }

    public init(host: any BrowserGPUComputeHost) { self.host = host }

    public func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {
        try await host.allocateBuffer(id: id, descriptor: descriptor)
    }
    public func destroyBuffer(id: ResourceID) async { await host.destroyBuffer(id: id) }
    public func close(owner: UInt64) async { await host.close(owner: owner) }
    public func submit(_ submission: Submission) async throws { try await host.submit(submission) }

    public func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) async throws {
        guard case .text(.wgsl, _) = shader.artifact else {
            throw GPUError.validation("Browser requires a compiled WGSL compute shader")
        }
        try await host.allocateComputePipeline(pipeline, shader: shader)
    }
    public func destroyComputePipeline(id: ResourceID) async { await host.destroyComputePipeline(id: id) }
    public func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        try await host.submitCommands(submission, commands: commands)
    }

    public func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws {
        guard let render = host as? any BrowserGPURenderHost else {
            throw GPUError.unsupported("Browser host does not implement textures")
        }
        try await render.allocateTexture(id: id, descriptor: descriptor)
    }
    public func destroyTexture(id: ResourceID) async {
        await (host as? any BrowserGPURenderHost)?.destroyTexture(id: id)
    }
    public func allocateRenderPipeline(
        _ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact
    ) async throws {
        guard let render = host as? any BrowserGPURenderHost else {
            throw GPUError.unsupported("Browser host does not implement render pipelines")
        }
        guard case .text(.wgsl, _) = vertex.artifact, case .text(.wgsl, _) = fragment.artifact else {
            throw GPUError.validation("Browser requires compiled WGSL vertex and fragment stages")
        }
        try await render.allocateRenderPipeline(pipeline, vertex: vertex, fragment: fragment)
    }
    public func destroyRenderPipeline(id: ResourceID) async {
        await (host as? any BrowserGPURenderHost)?.destroyRenderPipeline(id: id)
    }
}
