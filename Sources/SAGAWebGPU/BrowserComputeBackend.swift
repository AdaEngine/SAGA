import SAGATypes
import SAGABackend
import SAGAShaderIR

/// Browser implementation of the compute and buffer SPI, injected by the JS host boundary.
public protocol BrowserGPUComputeHost: BrowserGPUHost, ComputeBackendDevice {}

public protocol BrowserGPURenderHost: BrowserGPUComputeHost, RenderBackendDevice {}
public protocol BrowserGPUPresentationHost: BrowserGPURenderHost, PresentationBackendDevice {}

private func requireBrowserBindings(
    _ textures: [ComputeShaderArtifact.SampledTextureBinding],
    _ storage: [ComputeShaderArtifact.StorageTextureBinding],
    _ samplers: [ComputeShaderArtifact.SamplerBinding],
    _ storageBuffers: [ComputeShaderArtifact.StorageBinding],
    _ uniforms: [ComputeShaderArtifact.UniformBinding]
) throws {
    guard !storage.contains(where: { $0.access.contains(.atomic) || $0.format == .r64Uint }) else {
        throw GPUError.unsupported("Atomic and R64Uint storage textures require a native backend")
    }
    guard !textures.contains(where: { $0.bindingArray != nil && $0.bindingArray?.webBindingBase == nil }),
          !storage.contains(where: { $0.bindingArray != nil && $0.bindingArray?.webBindingBase == nil }),
          !samplers.contains(where: { $0.bindingArray != nil && $0.bindingArray?.webBindingBase == nil }),
          !storageBuffers.contains(where: { $0.bindingArray != nil && $0.bindingArray?.webBindingBase == nil }),
          !uniforms.contains(where: { $0.bindingArray != nil && $0.bindingArray?.webBindingBase == nil }) else {
        throw GPUError.unsupported("wgpu_binding_array is a native WGSL extension")
    }
}

public struct BrowserComputeBackend: ComputeBackendDevice, PresentationBackendDevice, SamplerBackendDevice, TextureViewBackendDevice,
                                     QuerySetBackendDevice, BufferMappingBackendDevice, ExternalTextureBackendDevice {
    private let host: any BrowserGPUComputeHost
    public var capabilities: DeviceCapabilities { host.capabilities }
    public var supportsExternalTextures: Bool {
        (host as? any ExternalTextureBackendDevice)?.supportsExternalTextures == true
    }
    public var supportsPresentation: Bool {
        (host as? any BrowserGPUPresentationHost)?.supportsPresentation == true
    }

    public init(host: any BrowserGPUComputeHost) { self.host = host }

    public func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {
        if descriptor.mappedAtCreation { _ = try mappingHost() }
        try await host.allocateBuffer(id: id, descriptor: descriptor)
    }
    public func destroyBuffer(id: ResourceID) async { await host.destroyBuffer(id: id) }
    public func allocateExternalTexture(id: ResourceID, descriptor: ExternalTextureDescriptor) async throws {
        guard let external = host as? any ExternalTextureBackendDevice, external.supportsExternalTextures else {
            throw GPUError.unsupported("Browser host does not support external textures")
        }
        try await external.allocateExternalTexture(id: id, descriptor: descriptor)
    }
    public func destroyExternalTexture(id: ResourceID) async {
        await (host as? any ExternalTextureBackendDevice)?.destroyExternalTexture(id: id)
    }
    public func beginBufferMapping(id: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64) async throws {
        try await mappingHost().beginBufferMapping(id: id, mode: mode, offset: offset, size: size)
    }
    public func readMappedBytes(id: ResourceID, offset: UInt64, size: UInt64) async throws -> [UInt8] {
        try await mappingHost().readMappedBytes(id: id, offset: offset, size: size)
    }
    public func writeMappedBytes(id: ResourceID, offset: UInt64, bytes: [UInt8]) async throws {
        try await mappingHost().writeMappedBytes(id: id, offset: offset, bytes: bytes)
    }
    public func endBufferMapping(id: ResourceID) async throws {
        try await mappingHost().endBufferMapping(id: id)
    }
    private func mappingHost() throws -> any BufferMappingBackendDevice {
        guard let mapping = host as? any BufferMappingBackendDevice else {
            throw GPUError.unsupported("Browser host does not implement buffer mapping")
        }
        return mapping
    }
    public func allocateQuerySet(id: ResourceID, descriptor: QuerySetDescriptor) async throws {
        try descriptor.validate(against: capabilities)
        guard let queryHost = host as? any QuerySetBackendDevice else {
            throw GPUError.unsupported("Browser host does not implement query sets")
        }
        try await queryHost.allocateQuerySet(id: id, descriptor: descriptor)
    }
    public func destroyQuerySet(id: ResourceID) async {
        await (host as? any QuerySetBackendDevice)?.destroyQuerySet(id: id)
    }
    public func close(owner: UInt64) async { await host.close(owner: owner) }
    public func submit(_ submission: Submission) async throws { try await host.submit(submission) }

    public func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) async throws {
        guard !shader.requiresDebugPrintf else { throw GPUError.unsupported("Browser WebGPU does not support shader debugPrintf") }
        guard !shader.requiresShaderInt16 else {
            throw GPUError.unsupported("Browser WebGPU does not support native 16-bit shader integers")
        }
        guard case .text(.wgsl, _) = shader.artifact else {
            throw GPUError.validation("Browser requires a compiled WGSL compute shader")
        }
        let size = shader.workgroupSize
        try capabilities.requireComputeWorkgroupSize(x: size.x, y: size.y, z: size.z)
        try capabilities.requireWorkgroupStorageSize(shader.workgroupStorageByteSize)
        guard shader.externalTextures.isEmpty || supportsExternalTextures else {
            throw GPUError.unsupported("Browser host does not support external textures")
        }
        try requireBrowserBindings(shader.sampledTextures, shader.storageTextures, shader.samplerBindings,
                                   shader.storageBindings, shader.uniformBindings)
        try capabilities.require(shader.storageTextureFeatures)
        try capabilities.requireImmediateSize(pipeline.immediateByteSize)
        if shader.requiresShaderInt64 { try capabilities.require([.shaderInt64]) }
        try capabilities.requireShaderInt64Atomics(minMax: shader.requiresShaderInt64AtomicMinMax,
                                                   allOps: shader.requiresShaderInt64AtomicAllOps)
        if shader.requiresSubgroups { try capabilities.require([.subgroups]) }
        if shader.requiresSubgroupUniformity { try capabilities.require([.subgroupUniformity]) }
        try capabilities.requireSubgroupSize(shader.requiredSubgroupSize)
        if shader.requiresSubgroupID { try capabilities.require([.subgroupID]) }
        if shader.requiresLinearIndexing { try capabilities.require([.linearIndexing]) }
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
        guard descriptor.sampleCount == 1 || descriptor.arrayLayerCount == 1 else {
            throw GPUError.unsupported("Browser WebGPU does not support multisampled texture arrays")
        }
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
        try await allocateRenderPipeline(pipeline, vertex: vertex, fragment: Optional(fragment))
    }

    public func allocateRenderPipeline(
        _ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact?
    ) async throws {
        guard !vertex.requiresDebugPrintf, fragment?.requiresDebugPrintf != true else {
            throw GPUError.unsupported("Browser WebGPU does not support shader debugPrintf")
        }
        guard !vertex.usesVertexWritableStorage else {
            throw GPUError.unsupported("Browser WebGPU does not support vertex writable storage")
        }
        guard !vertex.requiresShaderInt16, fragment?.requiresShaderInt16 != true else {
            throw GPUError.unsupported("Browser WebGPU does not support native 16-bit shader integers")
        }
        guard pipeline.multiviewMask == nil else { throw GPUError.unsupported("Browser WebGPU does not support multiview") }
        guard let render = host as? any BrowserGPURenderHost else {
            throw GPUError.unsupported("Browser host does not implement render pipelines")
        }
        guard vertex.stage == .vertex, case .text(.wgsl, _) = vertex.artifact else {
            throw GPUError.validation("Browser requires compiled WGSL vertex and fragment stages")
        }
        if let fragment {
            guard fragment.stage == .fragment, case .text(.wgsl, _) = fragment.artifact else {
                throw GPUError.validation("Browser requires compiled WGSL vertex and fragment stages")
            }
            guard !fragment.requiresPerVertex else {
                throw GPUError.unsupported("Browser WebGPU does not support per-vertex fragment inputs")
            }
        }
        guard (vertex.externalTextures.isEmpty && fragment?.externalTextures.isEmpty != false) || supportsExternalTextures else {
            throw GPUError.unsupported("Browser host does not support external textures")
        }
        for stage in [vertex, fragment].compactMap({ $0 }) {
            try requireBrowserBindings(stage.sampledTextures, stage.storageTextures, stage.samplerBindings,
                                       stage.storageBindings, stage.uniformBindings)
        }
        _ = try VertexBufferLayout.validate(pipeline.vertexBuffers, against: capabilities)
        try RenderPipeline.validateColorTargetLimits(pipeline.colorTargets, fragment: fragment, against: capabilities)
        try capabilities.validateInterstageInterfaces(vertex: vertex, fragment: fragment, topology: pipeline.primitive.topology)
        if fragment?.readsPrimitiveIndex == true { try capabilities.require([.primitiveIndex]) }
        if fragment?.requiresBarycentrics == true { try capabilities.require([.shaderBarycentrics]) }
        if fragment?.earlyDepthTest != nil { try capabilities.require([.shaderEarlyDepthTest]) }
        try capabilities.requireImmediateSize(pipeline.immediateByteSize)
        if vertex.requiresShaderInt64 || fragment?.requiresShaderInt64 == true { try capabilities.require([.shaderInt64]) }
        try capabilities.requireShaderInt64Atomics(
            minMax: vertex.requiresShaderInt64AtomicMinMax || fragment?.requiresShaderInt64AtomicMinMax == true,
            allOps: vertex.requiresShaderInt64AtomicAllOps || fragment?.requiresShaderInt64AtomicAllOps == true)
        if fragment?.requiresSubgroups == true || vertex.requiresSubgroups { try capabilities.require([.subgroups]) }
        if vertex.requiresSubgroupUniformity || fragment?.requiresSubgroupUniformity == true {
            try capabilities.require([.subgroupUniformity])
        }
        if fragment?.usesDualSourceBlending == true || pipeline.colorTargets.values.contains(where: { $0.colorState.usesSecondBlendSource }) {
            try capabilities.require([.dualSourceBlending])
        }
        if pipeline.depthState?.format == .depth32FloatStencil8 { try capabilities.require([.depth32FloatStencil8]) }
        try capabilities.require(Set(pipeline.colorTargets.values.compactMap { $0.format.hdrRenderFeature }))
        if pipeline.colorTargets.values.contains(where: { $0.format.isSnorm8Color }) {
            try capabilities.require([.snorm8Renderable])
        }
        if pipeline.colorTargets.values.contains(where: { $0.format.isNormalized16Color }) {
            try capabilities.require([.normalized16Textures])
        }
        if pipeline.sampleCount > 1 && pipeline.colorTargets.values.contains(where: {
            !$0.format.supportsMultisampledAttachment(in: capabilities)
        }) {
            throw GPUError.validation("Color attachment format does not support multisampling")
        }
        if pipeline.colorTargets.values.contains(where: { $0.format.isFloat32Color && $0.colorState.blend != nil }) {
            try capabilities.require([.float32Blendable])
        }
        if pipeline.primitive.unclippedDepth { try capabilities.require([.depthClipControl]) }
        let features = vertex.storageTextures.requiredStorageTextureFeatures
            .union((fragment?.storageTextures ?? []).requiredStorageTextureFeatures)
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
