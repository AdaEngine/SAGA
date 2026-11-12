import Testing
import SAGATypes
import SAGABackend
import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGAMetal

actor RenderTestBackend: PresentationBackendDevice {
    nonisolated let capabilities: DeviceCapabilities
    let started = Gate()
    let release = Gate()
    let allocationStarted = Gate()
    let allocationGate: Gate?
    let presentStarted = Gate()
    let presentationGate: Gate?
    nonisolated let supportsPresentation: Bool
    private(set) var submitted = 0
    private(set) var allocatedRenderPipelines = 0
    private(set) var textures: Set<ResourceID> = []
    private(set) var presented: [ResourceID] = []

    init(allocationGate: Gate? = nil, presentationGate: Gate? = nil,
         capabilities: DeviceCapabilities = try! .init(maxBufferSize: 4096, maxTextureDimension2D: 32)) {
        self.capabilities = capabilities
        self.allocationGate = allocationGate
        self.presentationGate = presentationGate
        self.supportsPresentation = presentationGate != nil
    }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) {}
    func destroyBuffer(id: ResourceID) {}
    func submit(_ submission: Submission) {}
    func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async {
        textures.insert(id)
        await allocationStarted.release()
        if let allocationGate { await allocationGate.wait() }
    }
    func destroyTexture(id: ResourceID) { textures.remove(id) }
    func allocateRenderPipeline(_ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact) {
        allocatedRenderPipelines += 1
    }
    func destroyRenderPipeline(id: ResourceID) {}
    func present(_ submission: Submission, source: ResourceID) async {
        presented.append(source)
        await presentStarted.release()
        if let presentationGate { await presentationGate.wait() }
    }
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        submitted += 1
        await started.release()
        await release.wait()
        return []
    }
}

private func makePipeline(_ device: Device, format: TextureFormat = .rgba8Unorm) async throws -> RenderPipeline {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(triangleProgram), entryPoint: "vertexMain", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl(triangleProgram), entryPoint: "fragmentMain", to: .msl)
    return try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: format)
}

@Test func singleSurfacePresentationRejectsAnUnknownSurfaceID() async throws {
    let backend = RenderTestBackend(presentationGate: Gate())
    let device = try await Device.make(backend: backend)
    let frame = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .copySource))
    let ticket = try await device.present(frame, surface: 1)
    await #expect(throws: GPUError.unsupported("Backend does not have the requested surface")) {
        try await ticket.wait()
    }
    #expect(await backend.presented.isEmpty)
    await device.close()
}

@Test func cancelledTextureAllocationRetiresUnpublishedResource() async throws {
    let gate = Gate()
    let backend = RenderTestBackend(allocationGate: gate)
    let device = try await Device.make(backend: backend)
    let task = Task {
        try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .copySource))
    }
    await backend.allocationStarted.wait()
    #expect(await backend.textures.count == 1)
    task.cancel()
    await gate.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await backend.textures.isEmpty)
    await device.close()
}

@Test func renderPipelineRejectsSwappedStagesBeforeBackendAllocation() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(triangleProgram), entryPoint: "vertexMain", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl(triangleProgram), entryPoint: "fragmentMain", to: .msl)
    await #expect(throws: GPUError.validation("Render pipeline requires vertex and fragment stages")) {
        try await device.createRenderPipeline(vertex: fragment, fragment: vertex, colorFormat: .rgba8Unorm)
    }
    await device.close()
}

@Test func renderPipelineValidatesVaryingLocationWidthAndSampling() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(varyingProgram), entryPoint: "vertexMain", to: .msl)
    for input in ["@location(3) v: vec4f", "@location(2) v: f32", "@location(5) v: f32"] {
        let fragment = try compiler.compileRenderStage(
            .wgsl("@fragment fn run(\(input)) -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }"),
            entryPoint: "run", to: .msl)
        await #expect(throws: GPUError.validation("Fragment inputs do not match vertex outputs")) {
            try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        }
    }
    await device.close()
}

@Test func renderCommandsRejectInvalidResourcesAndPinAcceptedTextures() async throws {
    let backend = RenderTestBackend()
    let device = try await Device.make(backend: backend)
    let other = try await Device.make(backend: backend)
    let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let copyOnly = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .copySource))
    let renderOnly = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .renderAttachment))
    let pipeline = try await makePipeline(device)
    let foreign = try await makePipeline(other)
    let bgra = try await makePipeline(device, format: .bgra8Unorm)
    let read = try await device.createBuffer(.init(size: 2048, usage: [.mapRead, .copyDestination]))
    let small = try await device.createBuffer(.init(size: 1820, usage: [.mapRead, .copyDestination]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    await #expect(throws: GPUError.self) {
        try await device.createTexture(.init(width: 33, height: 1, format: .rgba8Unorm, usage: .copySource))
    }
    await #expect(throws: GPUError.self) { try await device.destroyBuffer(texture) }
    await #expect(throws: GPUError.self) { try await device.destroyTexture(read) }
    for invalid in [
        BufferCommand.render(pipeline: foreign, target: texture, clear: clear, vertexCount: 3),
        .render(pipeline: bgra, target: texture, clear: clear, vertexCount: 3),
        .render(pipeline: pipeline, target: copyOnly, clear: clear, vertexCount: 3),
        .render(pipeline: pipeline, target: read, clear: clear, vertexCount: 3),
        .render(pipeline: pipeline, target: texture,
                clear: .init(red: .nan, green: 0, blue: 0, alpha: 1), vertexCount: 3),
        .copyTextureToBuffer(source: renderOnly, destination: read, bytesPerRow: 256),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 32),
        .copyTextureToBuffer(source: texture, destination: small, bytesPerRow: 256),
        .read(buffer: texture, byteCount: 4),
    ] {
        await #expect(throws: GPUError.self) { try await device.submitCommands([invalid]) }
    }
    #expect(await backend.submitted == 0)
    let ticket = try await device.submitCommands([
        .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
    ])
    #expect(ticket.sequence == 1)
    await backend.started.wait()
    try await device.destroyTexture(texture)
    #expect(await backend.textures.contains(texture))
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256)])
    }
    await backend.release.release()
    _ = try await ticket.wait()
    #expect(!(await backend.textures.contains(texture)))
    await device.close()
    await other.close()
    #expect(await backend.textures.isEmpty)
}

#if canImport(Metal)
@Test(arguments: [TextureFormat.rgba8Unorm, .bgra8Unorm])
func rendersAndReadsTextureThroughSAGAQueueOnMetal(format: TextureFormat) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await makePipeline(device, format: format)
        let texture = try await device.createTexture(.init(width: 8, height: 8, format: format,
                                                           usage: [.renderAttachment, .copySource]))
        let byteCount: UInt64 = 1824
        let buffer = try await device.createBuffer(.init(size: byteCount, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
            .read(buffer: buffer, byteCount: byteCount),
            .render(pipeline: pipeline, target: texture, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3),
            .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
            .read(buffer: buffer, byteCount: byteCount),
        ])
        try await device.destroyTexture(texture)
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2 && snapshots.allSatisfy { $0.count == Int(byteCount / 4) })
        #expect(snapshots[0].allSatisfy { $0 == 0 })
        let blue: UInt32 = format == .rgba8Unorm ? 0xffff0000 : 0xff0000ff
        let wrong = (0..<64).filter { index in
            let y = index / 8, x = index % 8
            guard x + y != 7 else { return false }
            let expected: UInt32 = x + y < 7 ? 0xff00ff00 : blue
            return snapshots[1][y * 64 + x] != expected
        }
        #expect(wrong.isEmpty)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
