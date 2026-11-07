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
    private let backend: any BackendDevice
    private enum ResourceKind: Equatable { case buffer, texture }
    private enum ResourceDescriptor {
        case buffer(BufferDescriptor)
        case texture(TextureDescriptor)

        var kind: ResourceKind {
            switch self { case .buffer: .buffer; case .texture: .texture }
        }
        var buffer: BufferDescriptor? {
            if case .buffer(let value) = self { return value }
            return nil
        }
        var texture: TextureDescriptor? {
            if case .texture(let value) = self { return value }
            return nil
        }
    }
    private struct Resource {
        let descriptor: ResourceDescriptor
        var ready = false
        var destroyed = false
        var uses = 0
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
        try await createResource(.buffer(descriptor))
    }

    public func createTexture(_ descriptor: TextureDescriptor) async throws -> ResourceID {
        try await createResource(.texture(descriptor))
    }

    private func createResource(_ descriptor: ResourceDescriptor) async throws -> ResourceID {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        do {
            try Task.checkCancellation()
            try requireActive()
            let allocate: (ResourceID) async throws -> Void
            switch descriptor {
            case .buffer(let value):
                try value.validate(against: capabilities)
                guard !value.mappedAtCreation else { throw GPUError.unsupported("Mapped allocation requires SAGA-4 mapping") }
                let backend = backend
                allocate = { try await backend.allocateBuffer(id: $0, descriptor: value) }
            case .texture(let value):
                try value.validate(against: capabilities)
                guard let render = backend as? any RenderBackendDevice else {
                    throw GPUError.unsupported("Backend does not implement textures")
                }
                allocate = { try await render.allocateTexture(id: $0, descriptor: value) }
            }
            guard nextResource < UInt64.max else { throw GPUError.outOfMemory }
            nextResource += 1
            let id = ResourceID(device: identity, serial: nextResource)
            resources[id] = Resource(descriptor: descriptor)
            do {
                try await allocate(id)
                try Task.checkCancellation()
                try requireActive()
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
    }

    public func validateResource(_ id: ResourceID) throws {
        do { try requireActive(); try requireResource(id) }
        catch { let error = normalize(error); record(error); throw error }
    }

    public func createComputePipeline(_ shader: ComputeShaderArtifact) async throws -> ComputePipeline {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        do {
            try Task.checkCancellation()
            try requireActive()
            guard let compute = backend as? any ComputeBackendDevice else {
                throw GPUError.unsupported("Backend does not implement compute pipelines")
            }
            guard nextResource < UInt64.max else { throw GPUError.outOfMemory }
            nextResource += 1
            let pipeline = ComputePipeline(
                id: ResourceID(device: identity, serial: nextResource),
                storageBindings: shader.storageBindings.map {
                    .init(slot: .init(group: $0.binding.group, binding: $0.binding.binding),
                          minimumByteSize: $0.minimumByteSize, isReadOnly: $0.isReadOnly)
                })
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

    public func createRenderPipeline(
        vertex: RenderStageArtifact, fragment: RenderStageArtifact, colorFormat: TextureFormat,
        vertexBuffers: [VertexBufferLayout] = []
    ) async throws -> RenderPipeline {
        let operationScopes = beginOperation()
        defer { endOperation(operationScopes) }
        do {
            try Task.checkCancellation()
            try requireActive()
            guard vertex.stage == .vertex, fragment.stage == .fragment else {
                throw GPUError.validation("Render pipeline requires vertex and fragment stages")
            }
            guard fragment.varyings.allSatisfy(vertex.varyings.contains) else {
                throw GPUError.validation("Fragment inputs do not match vertex outputs")
            }
            let formats = try VertexBufferLayout.validate(vertexBuffers)
            guard vertex.vertexInputs.allSatisfy({ input in
                guard let format = formats[input.location], input.scalar.width == 4 else { return false }
                switch (input.scalar.kind, format.dataType) {
                case (.float, .float), (.sint, .sint), (.uint, .uint): return true
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
            let pipeline = RenderPipeline(id: .init(device: identity, serial: nextResource), colorFormat: colorFormat,
                                          vertexBuffers: vertexBuffers)
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
        if resource.uses == 0 {
            resources.removeValue(forKey: id)
            await retire(id, kind: resource.descriptor.kind)
        }
    }

    private func retire(_ id: ResourceID, kind: ResourceKind) async {
        switch kind {
        case .buffer: await backend.destroyBuffer(id: id)
        case .texture: await (backend as? any RenderBackendDevice)?.destroyTexture(id: id)
        }
    }

    public func submitResourceUse(_ ids: [ResourceID]) throws -> SubmissionTicket {
        let backend = backend
        let (sequence, completion) = try enqueue(ids) { submission in
            try await backend.submit(submission)
        }
        return SubmissionTicket(sequence: sequence, completion: completion)
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
            for command in commands {
                for id in command.resources { try requireResource(id) }
                try command.validate(device: identity, descriptors: descriptors, textures: textures)
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
            for id in ids { try requireResource(id) }
            guard nextSubmission < UInt64.max else { throw GPUError.outOfMemory }
            nextSubmission += 1
            let unique = Array(Set(ids)).sorted { $0.serial < $1.serial }
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
