import SAGATypes
import SAGABackend
import SAGAShaderIR

/// Browser implementation of the compute and buffer SPI, injected by the JS host boundary.
public protocol BrowserGPUComputeHost: BrowserGPUHost, ComputeBackendDevice {}

public protocol BrowserGPURenderHost: BrowserGPUComputeHost, RenderBackendDevice {}
public protocol BrowserGPUPresentationHost: BrowserGPURenderHost, PresentationBackendDevice {}

public struct BrowserComputeBackend: ComputeBackendDevice, PresentationBackendDevice, SamplerBackendDevice, TextureViewBackendDevice {
    private let host: any BrowserGPUComputeHost
    public var capabilities: DeviceCapabilities { host.capabilities }
    public var supportsPresentation: Bool {
        (host as? any BrowserGPUPresentationHost)?.supportsPresentation == true
    }

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
        try capabilities.require(shader.storageTextureFeatures)
        try await host.allocateComputePipeline(pipeline, shader: shader)
    }
    public func destroyComputePipeline(id: ResourceID) async { await host.destroyComputePipeline(id: id) }
    public func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        try await host.submitCommands(submission, commands: commands)
    }

    public func allocateSampler(id: ResourceID, descriptor: SamplerDescriptor) async throws {
        guard let samplerHost = host as? any SamplerBackendDevice else {
            throw GPUError.unsupported("Browser host does not implement samplers")
        }
        try await samplerHost.allocateSampler(id: id, descriptor: descriptor)
    }
    public func destroySampler(id: ResourceID) async {
        await (host as? any SamplerBackendDevice)?.destroySampler(id: id)
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
    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>) async throws {
        try await allocateTextureView(id: id, texture: texture, mipLevels: mipLevels, aspect: .all)
    }
    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>,
                                    aspect: TextureViewAspect) async throws {
        guard let renderer = host as? any TextureViewBackendDevice else {
            throw GPUError.unsupported("Browser host does not implement texture views")
        }
        try await renderer.allocateTextureView(id: id, texture: texture, mipLevels: mipLevels, aspect: aspect)
    }
    public func allocateTextureView(id: ResourceID, texture: ResourceID, descriptor: TextureViewDescriptor) async throws {
        guard let renderer = host as? any TextureViewBackendDevice else {
            throw GPUError.unsupported("Browser host does not implement texture views")
        }
        try await renderer.allocateTextureView(id: id, texture: texture, descriptor: descriptor)
    }
    public func destroyTextureView(id: ResourceID) async {
        await (host as? any TextureViewBackendDevice)?.destroyTextureView(id: id)
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
        let features = vertex.storageTextures.requiredStorageTextureFeatures
            .union(fragment.storageTextures.requiredStorageTextureFeatures)
        try capabilities.require(features)
        try await render.allocateRenderPipeline(pipeline, vertex: vertex, fragment: fragment)
    }
    public func destroyRenderPipeline(id: ResourceID) async {
        await (host as? any BrowserGPURenderHost)?.destroyRenderPipeline(id: id)
    }

    public func present(_ submission: Submission, source: ResourceID) async throws {
        guard let presentation = host as? any BrowserGPUPresentationHost, presentation.supportsPresentation else {
            throw GPUError.unsupported("Browser host does not have a presentation surface")
        }
        try await presentation.present(submission, source: source)
    }
}
