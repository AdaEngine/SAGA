import SAGATypes
import SAGABackend
import SAGAShaderIR

private actor DeviceIdentities {
    static let shared = DeviceIdentities()
    private var next: UInt64 = 0
    func allocate() throws -> UInt64 {
        guard next < UInt64.max else { throw GPUError.outOfMemory }
        next += 1
        return next
    }
}

public struct ErrorScope: Hashable, Sendable {
    fileprivate let device: UInt64
    fileprivate let serial: UInt64
}

public struct SubmissionTicket: Sendable {
    public let sequence: UInt64
    fileprivate let completion: Task<Void, Error>

    /// Cancelling a waiter does not cancel already accepted GPU work.
    public func wait() async throws {
        try Task.checkCancellation()
        try await completion.value
        try Task.checkCancellation()
    }
}

public struct CommandTicket: Sendable {
    public let sequence: UInt64
    fileprivate let completion: Task<[[UInt32]], Error>

    public func wait() async throws -> [[UInt32]] {
        try Task.checkCancellation()
        let result = try await completion.value
        try Task.checkCancellation()
        return result
    }
}

public actor Device {
    public nonisolated let capabilities: DeviceCapabilities
    public nonisolated let identity: UInt64
    public nonisolated let supportsPresentation: Bool
    private let backend: any BackendDevice
    private enum ResourceKind: Equatable { case buffer, texture, externalTexture, sampler, textureView, querySet }
    private enum ResourceDescriptor {
        case buffer(BufferDescriptor)
        case texture(TextureDescriptor)
        case externalTexture(ExternalTextureDescriptor)
        case planarExternalTexture(PlanarExternalTextureDescriptor)
        case sampler(SamplerDescriptor)
        case textureView(TextureViewResource)
        case querySet(QuerySetDescriptor)

        var kind: ResourceKind {
            switch self {
            case .buffer: .buffer
            case .texture: .texture
            case .externalTexture, .planarExternalTexture: .externalTexture
            case .sampler: .sampler
            case .textureView: .textureView
            case .querySet: .querySet
            }
        }
        var buffer: BufferDescriptor? {
            if case .buffer(let value) = self { return value }
            return nil
        }
        var texture: TextureDescriptor? {
            if case .texture(let value) = self { return value }
            return nil
        }
        var sampler: SamplerDescriptor? {
            if case .sampler(let descriptor) = self { return descriptor }
            return nil
        }
        var textureView: TextureViewResource? {
            if case .textureView(let value) = self { return value }
            return nil
        }
        var externalPlanes: [ResourceID] {
            if case .planarExternalTexture(let descriptor) = self { return descriptor.planes }
            return []
        }
        var querySet: QuerySetDescriptor? {
            if case .querySet(let descriptor) = self { return descriptor }
            return nil
        }
    }
    private struct Resource {
        let descriptor: ResourceDescriptor
        var ready = false
        var destroyed = false
        var uses = 0
        var mapping: MappingSession?
    }
    private struct MappingSession {
        enum Phase { case queued, opening, mapped, closing }
        let handle: BufferMapping
        var phase: Phase
        var accessTail: Task<Void, Never>?
        var closeWaiters: [CheckedContinuation<Void, Error>] = []
    }
    private struct Scope {
        let token: ErrorScope
        let filter: ErrorFilter
        var error: GPUError?
        var pending = 0
        var popped = false
        var waiter: CheckedContinuation<GPUError?, Never>?
    }
    private var resources: [ResourceID: Resource] = [:]
    private var nextResource: UInt64 = 0
    private var nextMapping: UInt64 = 0
    private var nextSubmission: UInt64 = 0
    private var nextScope: UInt64 = 0
    private var scopes: [ErrorScope: Scope] = [:]
    private var activeScopes: [ErrorScope] = []
    private var pendingOperations = 0
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []
    private var uncaptured: [GPUError] = []
    private var loss: GPUError?
    private var tail: Task<Void, Error>?

    private init(identity: UInt64, backend: any BackendDevice) {
        self.identity = identity
        self.backend = backend
        self.capabilities = backend.capabilities
        self.supportsPresentation = (backend as? any PresentationBackendDevice)?.supportsPresentation == true
    }

    public static func make(backend: any BackendDevice, requiredFeatures: Set<GPUFeature> = []) async throws -> Device {
        try Task.checkCancellation()
        try backend.capabilities.require(requiredFeatures)
        let identity = try await DeviceIdentities.shared.allocate()
        try Task.checkCancellation()
        return Device(identity: identity, backend: backend)
    }

    private func requireActive() throws {
        if let loss { throw loss }
    }

    private func record(_ error: GPUError, in tokens: [ErrorScope]? = nil) {
        if let token = (tokens ?? activeScopes).reversed().first(where: { scopes[$0]?.filter.matches(error) == true }) {
            if scopes[token]?.error == nil { scopes[token]?.error = error }
        } else {
            uncaptured.append(error)
        }
    }

    private func beginOperation() -> [ErrorScope] {
        pendingOperations += 1
        for token in activeScopes { scopes[token]!.pending += 1 }
        return activeScopes
    }

    private func endOperation(_ tokens: [ErrorScope]) {
        for token in tokens {
            scopes[token]!.pending -= 1
            if let scope = scopes[token], scope.popped && scope.pending == 0 {
                scopes.removeValue(forKey: token)
                scope.waiter?.resume(returning: scope.error)
            }
        }
        pendingOperations -= 1
        if pendingOperations == 0 {
            let waiters = idleWaiters; idleWaiters.removeAll()
            for waiter in waiters { waiter.resume() }
        }
    }

    private func normalize(_ error: any Error) -> GPUError {
        (error as? GPUError) ?? .backendFailure(String(describing: error))
    }

    public func createBuffer(_ descriptor: BufferDescriptor) async throws -> ResourceID {
        try await createResource { .buffer(descriptor) }
    }

    public func createTexture(_ descriptor: TextureDescriptor) async throws -> ResourceID {
        try await createResource { .texture(descriptor) }
    }

    public func importExternalTexture(_ descriptor: ExternalTextureDescriptor) async throws -> ResourceID {
        try await createResource { .externalTexture(descriptor) }
    }

    public func importExternalTexture(_ descriptor: PlanarExternalTextureDescriptor) async throws -> ResourceID {
        try await createResource { .planarExternalTexture(descriptor) }
    }

    public func destroyExternalTexture(_ id: ResourceID) async throws {
        try await destroyResource(id, expected: .externalTexture)
    }

    public func createSampler(_ descriptor: SamplerDescriptor = .init()) async throws -> ResourceID {
        try await createResource { .sampler(descriptor) }
    }

    public func createQuerySet(_ descriptor: QuerySetDescriptor) async throws -> ResourceID {
        try await createResource { .querySet(descriptor) }
    }

    public func createTextureView(texture: ResourceID, _ descriptor: TextureViewDescriptor = .init()) async throws -> ResourceID {
        try await createResource {
            try requireResource(texture)
            guard let parent = resources[texture]?.descriptor.texture else {
                throw GPUError.validation("Texture view requires a texture resource")
            }
            return .textureView(.init(texture: texture, levels: try descriptor.mipLevels(in: parent), aspect: descriptor.aspect,
                                      layers: try descriptor.arrayLayers(in: parent), dimension: descriptor.resolvedDimension(in: parent)))
        }
    }

    private func createResource(_ describe: () throws -> ResourceDescriptor) async throws -> ResourceID {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        do {
            try Task.checkCancellation()
            try requireActive()
            let descriptor = try describe()
            let allocate: (ResourceID) async throws -> Void
            switch descriptor {
            case .buffer(let value):
                try value.validate(against: capabilities)
                if value.mappedAtCreation { _ = try mappingBackend() }
                let backend = backend
                allocate = { try await backend.allocateBuffer(id: $0, descriptor: value) }
            case .texture(let value):
                try value.validate(against: capabilities)
                guard let render = backend as? any RenderBackendDevice else {
                    throw GPUError.unsupported("Backend does not implement textures")
                }
                allocate = { try await render.allocateTexture(id: $0, descriptor: value) }
            case .externalTexture(let value):
                let external = try externalTextureBackend()
                guard value.videoFrame != 0 else { throw GPUError.validation("Invalid VideoFrame token") }
                allocate = { try await external.allocateExternalTexture(id: $0, descriptor: value) }
            case .planarExternalTexture(let value):
                guard let external = backend as? any PlanarExternalTextureBackendDevice, external.supportsExternalTextures else {
                    throw GPUError.unsupported("Backend does not support planar external textures")
                }
                try validateExternalPlanes(value)
                allocate = { try await external.allocatePlanarExternalTexture(id: $0, descriptor: value) }
            case .sampler(let value):
                guard let sampler = backend as? any SamplerBackendDevice else {
                    throw GPUError.unsupported("Backend does not implement samplers")
                }
                allocate = { try await sampler.allocateSampler(id: $0, descriptor: value) }
            case .textureView(let view):
                guard let renderer = backend as? any TextureViewBackendDevice else {
                    throw GPUError.unsupported("Backend does not implement texture views")
                }
                allocate = { try await renderer.allocateTextureView(id: $0, texture: view.texture, descriptor: view.descriptor) }
            case .querySet(let value):
                try value.validate(against: capabilities)
                guard let queryBackend = backend as? any QuerySetBackendDevice else {
                    throw GPUError.unsupported("Backend does not implement query sets")
                }
                allocate = { try await queryBackend.allocateQuerySet(id: $0, descriptor: value) }
            }
            guard nextResource < UInt64.max else { throw GPUError.outOfMemory }
            nextResource += 1
            let id = ResourceID(device: identity, serial: nextResource)
            resources[id] = Resource(descriptor: descriptor)
            do {
                try await allocate(id)
                try Task.checkCancellation()
                try requireActive()
                if let view = descriptor.textureView { try requireResource(view.texture) }
                for plane in descriptor.externalPlanes { try requireResource(plane) }
                if let buffer = descriptor.buffer, buffer.mappedAtCreation {
                    _ = try installMapping(id, mode: .write, offset: 0, size: buffer.size, phase: .mapped)
                }
                resources[id]?.ready = true
                return id
            } catch {
                resources.removeValue(forKey: id)
                await retire(id, kind: descriptor.kind)
                throw error
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let normalized = normalize(error)
            if case .deviceLost(let reason) = normalized { await lose(reason: reason) }
            else { record(normalized, in: operationScopes) }
            throw normalized
        }
    }

    private func requireResource(_ id: ResourceID) throws {
        guard id.device == identity, let resource = resources[id], resource.ready, !resource.destroyed else {
            throw GPUError.validation("Unknown, foreign or destroyed resource")
        }
        if let view = resource.descriptor.textureView,
           resources[view.texture]?.destroyed != false {
            throw GPUError.validation("Texture view has a destroyed parent texture")
        }
        for plane in resource.descriptor.externalPlanes { try requireResource(plane) }
    }

    private func validateExternalPlanes(_ descriptor: PlanarExternalTextureDescriptor) throws {
        guard descriptor.planes.count == descriptor.format.planeComponents.count else {
            throw GPUError.validation("Invalid external texture plane count or extent")
        }
        var first: (width: UInt32, height: UInt32)?
        for (index, plane) in descriptor.planes.enumerated() {
            try requireResource(plane)
            let view = resources[plane]?.descriptor.textureView
            guard let texture = resources[view?.texture ?? plane]?.descriptor.texture else {
                throw GPUError.validation("External texture planes must be textures or views")
            }
            let extent = try descriptor.planeExtent(at: index, texture: texture, view: view, capabilities: capabilities)
            if index == 0 { first = extent }
        }
        guard let first else { throw GPUError.validation("External texture requires a first plane") }
        _ = try descriptor.parameterWords(planeWidth: first.width, planeHeight: first.height)
    }

    private func requireGPUResource(_ id: ResourceID) throws {
        try requireResource(id)
        guard resources[id]?.mapping == nil else { throw GPUError.validation("Buffer is mapped or has a pending mapping") }
    }

    public func validateResource(_ id: ResourceID) throws {
        do { try requireActive(); try requireResource(id) }
        catch { let error = normalize(error); record(error); throw error }
    }

    /// Non-filtering slots accept nearest samplers and permit unfilterable float textures.
    public func createComputePipeline(
        _ shader: ComputeShaderArtifact, nonFilteringSamplers: Set<ComputeBindingSlot> = []
    ) async throws -> ComputePipeline {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        do {
            try Task.checkCancellation()
            try requireActive()
            let size = shader.workgroupSize
            try capabilities.requireComputeWorkgroupSize(x: size.x, y: size.y, z: size.z)
            try capabilities.requireWorkgroupStorageSize(shader.workgroupStorageByteSize)
            if !shader.externalTextures.isEmpty { _ = try externalTextureBackend() }
            if shader.requiresShaderF16 { try capabilities.require([.shaderF16]) }
            if shader.requiresShaderInt16 { try capabilities.require([.shaderInt16]) }
            if shader.requiresShaderInt64 { try capabilities.require([.shaderInt64]) }
            try capabilities.requireShaderInt64Atomics(minMax: shader.requiresShaderInt64AtomicMinMax,
                                                      allOps: shader.requiresShaderInt64AtomicAllOps)
            if shader.requiresSubgroups { try capabilities.require([.subgroups]) }
            if shader.requiresCooperativeMatrix { try capabilities.require([.cooperativeMatrix]) }
            if shader.requiresSubgroupUniformity { try capabilities.require([.subgroupUniformity]) }
            try capabilities.requireSubgroupSize(shader.requiredSubgroupSize)
            if shader.requiresSubgroupID { try capabilities.require([.subgroupID]) }
            if shader.requiresLinearIndexing { try capabilities.require([.linearIndexing]) }
            if shader.requiresUniformBufferStandardLayout {
                try capabilities.require([.uniformBufferStandardLayout])
            }
            guard let compute = backend as? any ComputeBackendDevice else {
                throw GPUError.unsupported("Backend does not implement compute pipelines")
            }
            guard nextResource < UInt64.max else { throw GPUError.outOfMemory }
            nextResource += 1
            let pipeline = try ComputePipeline(
                id: ResourceID(device: identity, serial: nextResource), shader: shader, nonFilteringSamplers: nonFilteringSamplers)
            try capabilities.requireImmediateSize(pipeline.immediateByteSize)
            do {
                try await compute.allocateComputePipeline(pipeline, shader: shader)
                try Task.checkCancellation()
                try requireActive()
                return pipeline
            } catch {
                await compute.destroyComputePipeline(id: pipeline.id)
                throw error
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let normalized = normalize(error)
            if case .deviceLost(let reason) = normalized { await lose(reason: reason) }
            else { record(normalized, in: operationScopes) }
            throw normalized
        }
    }

    /// Pass nil for the fragment stage to render to a depth or stencil attachment only.
    /// Non-filtering sampler slots apply to both shader stages.
    public func createRenderPipeline(
        vertex: RenderStageArtifact, fragment: RenderStageArtifact?, colorFormat: TextureFormat?,
        vertexBuffers: [VertexBufferLayout] = [], depthState: RenderDepthState? = nil, sampleCount: UInt32 = 1,
        sampleMask: UInt32 = .max, alphaToCoverageEnabled: Bool = false, primitive: RenderPrimitiveState = .init(),
        colorState: RenderColorState = .init(), additionalColorTargets: [UInt32: RenderColorTarget] = [:],
        nonFilteringSamplers: Set<ComputeBindingSlot> = [], multiviewMask: UInt32? = nil
    ) async throws -> RenderPipeline {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        do {
            try Task.checkCancellation()
            try requireActive()
            if !vertex.externalTextures.isEmpty || fragment?.externalTextures.isEmpty == false { _ = try externalTextureBackend() }
            if vertex.requiresShaderF16 || fragment?.requiresShaderF16 == true { try capabilities.require([.shaderF16]) }
            if vertex.requiresShaderInt16 || fragment?.requiresShaderInt16 == true { try capabilities.require([.shaderInt16]) }
            if vertex.requiresShaderInt64 || fragment?.requiresShaderInt64 == true { try capabilities.require([.shaderInt64]) }
            try capabilities.requireShaderInt64Atomics(
                minMax: vertex.requiresShaderInt64AtomicMinMax || fragment?.requiresShaderInt64AtomicMinMax == true,
                allOps: vertex.requiresShaderInt64AtomicAllOps || fragment?.requiresShaderInt64AtomicAllOps == true)
            if vertex.requiresSubgroups || fragment?.requiresSubgroups == true { try capabilities.require([.subgroups]) }
            if vertex.requiresSubgroupUniformity || fragment?.requiresSubgroupUniformity == true {
                try capabilities.require([.subgroupUniformity])
            }
            if fragment?.usesDualSourceBlending == true || (colorFormat != nil && colorState.usesSecondBlendSource) ||
                additionalColorTargets.values.contains(where: { $0.colorState.usesSecondBlendSource }) {
                try capabilities.require([.dualSourceBlending])
            }
            if fragment?.readsPrimitiveIndex == true { try capabilities.require([.primitiveIndex]) }
            if fragment?.requiresBarycentrics == true { try capabilities.require([.shaderBarycentrics]) }
            if fragment?.earlyDepthTest != nil { try capabilities.require([.shaderEarlyDepthTest]) }
            if primitive.unclippedDepth { try capabilities.require([.depthClipControl]) }
            if vertex.requiresUniformBufferStandardLayout || fragment?.requiresUniformBufferStandardLayout == true {
                try capabilities.require([.uniformBufferStandardLayout])
            }
            guard colorFormat?.isColorAttachment ?? true else {
                throw GPUError.validation("Render pipeline requires a supported color attachment format")
            }
            guard depthState?.format.isDepthStencil ?? true else {
                throw GPUError.validation("Render pipeline requires a supported depth format")
            }
            guard vertex.stage == .vertex, fragment.map({ $0.stage == .fragment }) ?? true else {
                throw GPUError.validation("Render pipeline requires vertex and fragment stages")
            }
            try capabilities.validateInterstageInterfaces(vertex: vertex, fragment: fragment, topology: primitive.topology,
                                                         multiviewMask: multiviewMask)
            let formats = try VertexBufferLayout.validate(vertexBuffers, against: capabilities)
            if let pulling = vertex.metalVertexPullingLayout, pulling != VertexBufferLayout.metalPullingSignature(vertexBuffers) {
                throw GPUError.validation("Vertex buffers do not match the Metal pulling layout")
            }
            guard vertex.vertexInputs.allSatisfy({ input in
                guard let format = formats[input.location] else { return false }
                switch (input.scalar, format.dataType) {
                case (.f16, .float), (.f32, .float), (.i32, .sint), (.u32, .uint): return true
                default: return false
                }
            }) else {
                throw GPUError.validation("Vertex layout does not provide compatible shader inputs")
            }
            guard let render = backend as? any RenderBackendDevice else {
                throw GPUError.unsupported("Backend does not implement render pipelines")
            }
            guard nextResource < UInt64.max else { throw GPUError.outOfMemory }
            nextResource += 1
            let pipeline = try RenderPipeline(id: .init(device: identity, serial: nextResource), colorFormat: colorFormat,
                                              vertexBuffers: vertexBuffers, depthState: depthState, sampleCount: sampleCount,
                                              sampleMask: sampleMask, alphaToCoverageEnabled: alphaToCoverageEnabled,
                                              primitive: primitive, colorState: colorState, additionalColorTargets: additionalColorTargets,
                                              vertex: vertex, fragment: fragment, nonFilteringSamplers: nonFilteringSamplers,
                                              capabilities: capabilities, multiviewMask: multiviewMask)
            do {
                try await render.allocateRenderPipeline(pipeline, vertex: vertex, fragment: fragment)
                try Task.checkCancellation()
                try requireActive()
                return pipeline
            } catch {
                await render.destroyRenderPipeline(id: pipeline.id)
                throw error
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let normalized = normalize(error)
            if case .deviceLost(let reason) = normalized { await lose(reason: reason) }
            else { record(normalized, in: operationScopes) }
            throw normalized
        }
    }

    public func destroyBuffer(_ id: ResourceID) async throws {
        try await destroyResource(id, expected: .buffer)
    }

    public func destroyTexture(_ id: ResourceID) async throws {
        try await destroyResource(id, expected: .texture)
    }

    public func destroySampler(_ id: ResourceID) async throws {
        try await destroyResource(id, expected: .sampler)
    }

    public func destroyTextureView(_ id: ResourceID) async throws {
        try await destroyResource(id, expected: .textureView)
    }

    public func destroyQuerySet(_ id: ResourceID) async throws {
        try await destroyResource(id, expected: .querySet)
    }

    private func destroyResource(_ id: ResourceID, expected: ResourceKind? = nil) async throws {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        guard id.device == identity, id.serial > 0, id.serial <= nextResource else {
            let error = GPUError.validation("Unknown or foreign resource")
            record(error, in: operationScopes); throw error
        }
        // Removed serials are retired forever; no per-destroy tombstone storage is needed.
        guard let resource = resources[id] else { return }
        if let expected, resource.descriptor.kind != expected {
            let error = GPUError.validation("Resource kind does not match destroy operation")
            record(error, in: operationScopes); throw error
        }
        guard resource.ready else {
            let error = GPUError.validation("Resource allocation has not completed")
            record(error, in: operationScopes); throw error
        }
        if resource.destroyed { return }
        resources[id]?.destroyed = true
        if let mapping = resource.mapping {
            await cancelMappingForDestroy(mapping)
            return
        }
        if resource.uses == 0 {
            resources.removeValue(forKey: id)
            await retire(id, kind: resource.descriptor.kind)
        }
    }

    private func retire(_ id: ResourceID, kind: ResourceKind) async {
        switch kind {
        case .buffer: await backend.destroyBuffer(id: id)
        case .texture: await (backend as? any RenderBackendDevice)?.destroyTexture(id: id)
        case .externalTexture: await (backend as? any ExternalTextureBackendDevice)?.destroyExternalTexture(id: id)
        case .sampler: await (backend as? any SamplerBackendDevice)?.destroySampler(id: id)
        case .textureView: await (backend as? any TextureViewBackendDevice)?.destroyTextureView(id: id)
        case .querySet: await (backend as? any QuerySetBackendDevice)?.destroyQuerySet(id: id)
        }
    }

    private func externalTextureBackend() throws -> any ExternalTextureBackendDevice {
        guard let external = backend as? any ExternalTextureBackendDevice, external.supportsExternalTextures else {
            throw GPUError.unsupported("Backend does not support external textures")
        }
        return external
    }

    public func submitResourceUse(_ ids: [ResourceID]) throws -> SubmissionTicket {
        let backend = backend
        let (sequence, completion) = try enqueue(ids) { submission in
            try await backend.submit(submission)
        }
        return SubmissionTicket(sequence: sequence, completion: completion)
    }

    /// Copy an offscreen color texture to the attached surface in submission order.
    public func present(_ source: ResourceID) throws -> SubmissionTicket {
        let presentation: any PresentationBackendDevice
        do {
            try Task.checkCancellation()
            try requireActive()
            guard supportsPresentation, let backend = backend as? any PresentationBackendDevice else {
                throw GPUError.unsupported("Backend does not have a presentation surface")
            }
            presentation = backend
            try requireResource(source)
            guard let texture = resources[source]?.descriptor.texture, texture.format.isNormalizedColor,
                  texture.sampleCount == 1, texture.dimension == .d2, texture.arrayLayerCount == 1,
                  texture.usage.contains(.copySource) else {
                throw GPUError.validation("Presentation requires a copy-source texture")
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let error = normalize(error); record(error); throw error
        }
        let (sequence, completion) = try enqueue([source]) { submission in
            try await presentation.present(submission, source: source)
        }
        return SubmissionTicket(sequence: sequence, completion: completion)
    }

    /// Enqueues a byte upload. Source offsets and counts are in bytes.
    public func writeBuffer(
        _ buffer: ResourceID, offset: UInt64 = 0, bytes: [UInt8], dataOffset: UInt64 = 0, byteCount: UInt64? = nil
    ) throws -> CommandTicket {
        try submitCommands([.writeBytes(buffer: buffer, offset: offset, bytes: bytes,
                                        dataOffset: dataOffset, byteCount: byteCount)])
    }

    public func writeTexture(
        _ destination: ResourceID, data: [UInt8], layout: TextureDataLayout = .init(),
        region: TextureCopyRegion, aspect: TextureViewAspect = .all
    ) throws -> CommandTicket {
        try submitCommands([.writeTexture(destination: destination, data: data, layout: layout,
                                          region: region, aspect: aspect)])
    }

    public func submitCommands(_ commands: [BufferCommand]) throws -> CommandTicket {
        let commandBackend: any CommandBackendDevice
        do {
            try Task.checkCancellation()
            try requireActive()
            guard let implementation = backend as? any CommandBackendDevice else {
                throw GPUError.unsupported("Backend does not implement command submission")
            }
            commandBackend = implementation
            let descriptors = resources.compactMapValues { $0.descriptor.buffer }
            let textures = resources.compactMapValues { $0.descriptor.texture }
            let textureViews = resources.compactMapValues { $0.descriptor.textureView }
            let externalTextures = Set(resources.compactMap { id, resource in
                resource.descriptor.kind == .externalTexture ? id : nil
            })
            let externalPlanes = resources.compactMapValues { resource in
                let planes = resource.descriptor.externalPlanes
                return planes.isEmpty ? nil : planes
            }
            let samplerDescriptors = resources.compactMapValues { $0.descriptor.sampler }
            let querySets = resources.compactMapValues { $0.descriptor.querySet }
            for command in commands {
                for id in command.resources { try requireGPUResource(id) }
                try command.validate(device: identity, descriptors: descriptors, textures: textures,
                                     samplerDescriptors: samplerDescriptors, textureViews: textureViews,
                                     maximumViewportDimension: capabilities.maxTextureDimension2D, querySets: querySets,
                                     maxMultiviewViewCount: capabilities.maxMultiviewViewCount,
                                     features: capabilities.features, externalTextures: externalTextures,
                                     externalPlanes: externalPlanes,
                                     maxComputeWorkgroupsPerDimension: capabilities.maxComputeWorkgroupsPerDimension)
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let error = normalize(error); record(error); throw error
        }
        let (sequence, completion) = try enqueue(commands.flatMap(\.resources)) { submission in
            try await commandBackend.submitCommands(submission, commands: commands)
        }
        return CommandTicket(sequence: sequence, completion: completion)
    }

    private func enqueue<Result: Sendable>(
        _ ids: [ResourceID], operation: @escaping @Sendable (Submission) async throws -> Result
    ) throws -> (UInt64, Task<Result, Error>) {
        do {
            try Task.checkCancellation()
            try requireActive()
            for id in ids { try requireGPUResource(id) }
            guard nextSubmission < UInt64.max else { throw GPUError.outOfMemory }
            nextSubmission += 1
            let planes = ids.flatMap { resources[$0]?.descriptor.externalPlanes ?? [] }
            let parents = (ids + planes).compactMap { resources[$0]?.descriptor.textureView?.texture }
            let unique = Set(ids).union(planes).union(parents).sorted { $0.serial < $1.serial }
            for id in unique { resources[id]!.uses += 1 }
            let submission = Submission(sequence: nextSubmission, resources: unique)
            let predecessor = tail
            let operationScopes = beginOperation()
            let completion = Task {
                defer { endOperation(operationScopes) }
                if let predecessor { _ = await predecessor.result }
                do {
                    try requireActive()
                    let result = try await operation(submission)
                    await finish(unique)
                    return result
                } catch {
                    let error = normalize(error)
                    if case .deviceLost(let reason) = error { await lose(reason: reason) }
                    else if error == .timeout { await lose(reason: "GPU completion timeout") }
                    else { record(error, in: operationScopes) }
                    await finish(unique)
                    throw error
                }
            }
            tail = Task { _ = try await completion.value }
            return (nextSubmission, completion)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let error = normalize(error); record(error); throw error
        }
    }

    private func finish(_ ids: [ResourceID]) async {
        var release: [ResourceID] = []
        for id in ids {
            resources[id]!.uses -= 1
            if resources[id]!.destroyed && resources[id]!.uses == 0 { release.append(id) }
        }
        for id in release {
            if let resource = resources.removeValue(forKey: id) {
                await retire(id, kind: resource.descriptor.kind)
            }
        }
    }

    public func pushErrorScope(_ filter: ErrorFilter) throws -> ErrorScope {
        guard nextScope < UInt64.max else { throw GPUError.outOfMemory }
        nextScope += 1
        let token = ErrorScope(device: identity, serial: nextScope)
        scopes[token] = Scope(token: token, filter: filter)
        activeScopes.append(token)
        return token
    }

    public func popErrorScope(_ token: ErrorScope) async throws -> GPUError? {
        guard activeScopes.last == token else { throw GPUError.validation("Error scopes must be popped LIFO on their device") }
        activeScopes.removeLast()
        scopes[token]!.popped = true
        if scopes[token]!.pending == 0 { return scopes.removeValue(forKey: token)!.error }
        return await withCheckedContinuation { scopes[token]!.waiter = $0 }
    }

    /// Explicit owner teardown. Backends must eventually complete outstanding operations, even on loss.
    public func close() async {
        await lose(reason: "Device closed")
        if pendingOperations > 0 {
            await withCheckedContinuation { idleWaiters.append($0) }
        }
        await backend.close(owner: identity)
    }

    public func takeUncapturedErrors() -> [GPUError] {
        defer { uncaptured.removeAll() }
        return uncaptured
    }

    public func lose(reason: String) async {
        guard loss == nil else { return }
        let teardownScopes = beginOperation()
        defer { endOperation(teardownScopes) }
        let error = GPUError.deviceLost(reason)
        loss = error
        record(error)
        let ready = resources.compactMap { $0.value.ready && !$0.value.destroyed ? $0.key : nil }
        for id in ready { try? await destroyResource(id) }
    }
}

extension Device {
    public func bufferMapState(_ buffer: ResourceID) throws -> BufferMapState {
        do {
            try requireActive()
            _ = try bufferDescriptor(buffer)
            switch resources[buffer]?.mapping?.phase {
            case .mapped: return .mapped
            case nil: return .unmapped
            default: return .pending
            }
        } catch {
            let error = normalize(error)
            record(error)
            throw error
        }
    }

    /// Returns the active handle, including for a buffer created mapped.
    public func mappedBuffer(_ buffer: ResourceID) throws -> BufferMapping {
        do {
            try requireActive()
            _ = try bufferDescriptor(buffer)
            guard let session = resources[buffer]?.mapping, session.phase == .mapped else {
                throw GPUError.validation("Buffer has no active mapping")
            }
            return session.handle
        } catch {
            let error = normalize(error)
            record(error)
            throw error
        }
    }

    /// Reserves the buffer immediately and opens the mapping after accepted GPU work retires.
    public func mapBuffer(
        _ buffer: ResourceID, mode: BufferMapMode, offset: UInt64 = 0, size: UInt64? = nil
    ) async throws -> BufferMapping {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        do {
            try Task.checkCancellation()
            try requireActive()
            let descriptor = try bufferDescriptor(buffer)
            let usage: BufferUsage = mode == .read ? .mapRead : .mapWrite
            guard resources[buffer]?.mapping == nil, descriptor.usage.contains(usage),
                  offset % 8 == 0, offset <= descriptor.size else {
                throw GPUError.validation("Invalid buffer mapping state, usage or offset")
            }
            let length = size ?? (descriptor.size - offset)
            guard length % 4 == 0, length <= descriptor.size - offset else {
                throw GPUError.validation("Invalid buffer mapping size")
            }
            let backend = try mappingBackend()
            let handle = try installMapping(buffer, mode: mode, offset: offset, size: length, phase: .queued)
            let predecessor = tail
            var opened = false
            do {
                if let predecessor { _ = await predecessor.result }
                try requireActive()
                try Task.checkCancellation()
                guard resources[buffer]?.mapping?.handle == handle,
                      resources[buffer]?.mapping?.phase == .queued else { throw CancellationError() }
                resources[buffer]?.mapping?.phase = .opening
                try await backend.beginBufferMapping(id: buffer, mode: mode, offset: offset, size: length)
                opened = true
                try requireActive()
                try Task.checkCancellation()
                guard resources[buffer]?.mapping?.phase == .opening else { throw CancellationError() }
                resources[buffer]?.mapping?.phase = .mapped
                return handle
            } catch {
                let cancelled = resources[buffer]?.mapping?.handle == handle &&
                    resources[buffer]?.mapping?.phase == .closing && loss == nil
                if opened {
                    do { try await backend.endBufferMapping(id: buffer) }
                    catch {
                        resources[buffer]?.destroyed = true
                        await finishMapping(handle, error: normalize(error))
                        throw error
                    }
                }
                await finishMapping(handle)
                if cancelled { throw CancellationError() }
                throw error
            }
        } catch {
            try await reportMappingError(error, in: operationScopes)
        }
    }

    /// Copies a byte range from the mapping. Offsets are relative to the buffer.
    public func readMappedBytes(
        _ mapping: BufferMapping, offset: UInt64? = nil, size: UInt64? = nil
    ) async throws -> [UInt8] {
        try await accessMappedBuffer(mapping, offset: offset ?? mapping.offset, size: size, writable: false) {
            backend, offset, length in
            let bytes = try await backend.readMappedBytes(id: mapping.buffer, offset: offset, size: length)
            guard bytes.count == Int(length) else { throw GPUError.backendFailure("Mapped readback has an invalid size") }
            return bytes
        }
    }

    public func writeMappedBytes(_ mapping: BufferMapping, offset: UInt64, bytes: [UInt8]) async throws {
        try await accessMappedBuffer(mapping, offset: offset, size: UInt64(bytes.count), writable: true) {
            backend, offset, _ in
            try await backend.writeMappedBytes(id: mapping.buffer, offset: offset, bytes: bytes)
        }
    }

    /// Publishes mapped writes or cancels a pending mapping.
    public func unmapBuffer(_ buffer: ResourceID) async throws {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        do {
            try requireActive()
            _ = try bufferDescriptor(buffer)
            guard let session = resources[buffer]?.mapping else { return }
            switch session.phase {
            case .queued:
                await finishMapping(session.handle)
            case .opening, .closing:
                resources[buffer]?.mapping?.phase = .closing
                try await withCheckedThrowingContinuation { continuation in
                    resources[buffer]!.mapping!.closeWaiters.append(continuation)
                }
            case .mapped:
                try await closeActiveMapping(session, backend: mappingBackend())
            }
        } catch {
            try await reportMappingError(error, in: operationScopes)
        }
    }

    private func bufferDescriptor(_ id: ResourceID) throws -> BufferDescriptor {
        try requireResource(id)
        guard let descriptor = resources[id]?.descriptor.buffer else {
            throw GPUError.validation("Mapping requires a buffer resource")
        }
        return descriptor
    }

    private func mappingBackend() throws -> any BufferMappingBackendDevice {
        guard let mapping = backend as? any BufferMappingBackendDevice else {
            throw GPUError.unsupported("Backend does not implement buffer mapping")
        }
        return mapping
    }

    private func installMapping(
        _ buffer: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64, phase: MappingSession.Phase
    ) throws -> BufferMapping {
        guard nextMapping < UInt64.max else { throw GPUError.outOfMemory }
        nextMapping += 1
        let handle = BufferMapping(buffer: buffer, mode: mode, offset: offset, size: size, generation: nextMapping)
        resources[buffer]!.uses += 1
        resources[buffer]!.mapping = MappingSession(handle: handle, phase: phase)
        return handle
    }

    private func accessMappedBuffer<Result: Sendable>(
        _ handle: BufferMapping, offset: UInt64, size: UInt64?, writable: Bool,
        operation: @escaping @Sendable (any BufferMappingBackendDevice, UInt64, UInt64) async throws -> Result
    ) async throws -> Result {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        do {
            try Task.checkCancellation()
            try requireActive()
            _ = try bufferDescriptor(handle.buffer)
            guard let session = resources[handle.buffer]?.mapping, session.handle == handle, session.phase == .mapped,
                  !writable || handle.mode == .write else {
                throw GPUError.validation("Invalid or read-only mapping")
            }
            let length = try handle.accessSize(offset: offset, size: size)
            let backend = try mappingBackend()
            let predecessor = session.accessTail
            let access = Task {
                if let predecessor { await predecessor.value }
                return try await operation(backend, offset, length)
            }
            resources[handle.buffer]?.mapping?.accessTail = Task { _ = await access.result }
            let result = try await access.value
            try requireActive()
            try Task.checkCancellation()
            return result
        } catch {
            try await reportMappingError(error, in: operationScopes)
        }
    }

    private func closeActiveMapping(_ session: MappingSession, backend: any BufferMappingBackendDevice) async throws {
        let id = session.handle.buffer
        resources[id]?.mapping?.phase = .closing
        if let predecessor = session.accessTail { await predecessor.value }
        do {
            try await backend.endBufferMapping(id: id)
        } catch {
            let error = normalize(error)
            if resources[id]?.destroyed == true {
                await finishMapping(session.handle, error: error)
            } else {
                let waiters = resources[id]?.mapping?.closeWaiters ?? []
                resources[id]?.mapping?.phase = .mapped
                resources[id]?.mapping?.closeWaiters.removeAll()
                for waiter in waiters { waiter.resume(throwing: error) }
            }
            throw error
        }
        await finishMapping(session.handle)
    }

    private func cancelMappingForDestroy(_ session: MappingSession) async {
        switch session.phase {
        case .queued:
            await finishMapping(session.handle)
        case .opening:
            resources[session.handle.buffer]?.mapping?.phase = .closing
        case .mapped:
            if let backend = backend as? any BufferMappingBackendDevice {
                try? await closeActiveMapping(session, backend: backend)
            }
        case .closing:
            break
        }
    }

    private func finishMapping(_ handle: BufferMapping, error: GPUError? = nil) async {
        guard let session = resources[handle.buffer]?.mapping, session.handle == handle else { return }
        resources[handle.buffer]?.mapping = nil
        await finish([handle.buffer])
        for waiter in session.closeWaiters {
            if let error { waiter.resume(throwing: error) }
            else { waiter.resume() }
        }
    }

    private func reportMappingError(_ error: any Error, in scopes: [ErrorScope]) async throws -> Never {
        if error is CancellationError { throw CancellationError() }
        let error = normalize(error)
        if case .deviceLost(let reason) = error { await lose(reason: reason) }
        else if error == .timeout { await lose(reason: "GPU buffer mapping timeout") }
        else { record(error, in: scopes) }
        throw error
    }
}
