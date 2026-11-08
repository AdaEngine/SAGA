import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAWebGPU
import SAGAMetal

private let timestampCompute = """
    @group(0) @binding(0) var<storage, read_write> result: array<u32, 64>;
    @compute @workgroup_size(64) fn run(@builtin(local_invocation_index) lane: u32) {
        result[lane] = lane * 7u;
    }
    """

private let timestampRender = """
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f { return vec4f(point, 0.5, 1.0); }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

private actor TimestampTestHost: BrowserGPUComputeHost, QuerySetBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024, features: [.timestampQuery])
    let started = Gate()
    let complete = Gate()
    private(set) var live: Set<ResourceID> = []
    private(set) var submits = 0

    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) {}
    func destroyBuffer(id: ResourceID) {}
    func allocateQuerySet(id: ResourceID, descriptor: QuerySetDescriptor) { live.insert(id) }
    func destroyQuerySet(id: ResourceID) { live.remove(id) }
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) {}
    func destroyComputePipeline(id: ResourceID) {}
    func submit(_ submission: Submission) {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        submits += 1
        await started.release()
        await complete.wait()
        return []
    }
}

@Test func timestampQueryValidationAndRetirementFollowQueueLifetime() async throws {
    let host = TimestampTestHost()
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    let unsupported = try await Device.make(backend: TestBackend())
    await #expect(throws: GPUError.validation("Feature not advertised")) {
        try await unsupported.createQuerySet(.init(type: .timestamp, count: 4))
    }
    await unsupported.close()
    for count: UInt32 in [0, 4097] {
        await #expect(throws: GPUError.self) { try await device.createQuerySet(.init(type: .timestamp, count: count)) }
    }
    let query = try await device.createQuerySet(.init(type: .timestamp, count: 4))
    let result = try await device.createBuffer(.init(size: 288, usage: [.queryResolve, .copySource]))
    let incorrectUsage = try await device.createBuffer(.init(size: 32, usage: .copySource))
    let compute = try IndependentShaderCompiler().compileCompute(.wgsl(timestampCompute), to: .wgsl)
    let pipeline = try await device.createComputePipeline(compute)
    let output = try await device.createBuffer(.init(size: 256, usage: .storage))
    let pass = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1))
    let timed = BufferCommand.timestamped(pass: pass, writes: .init(querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 2))
    for invalid: BufferCommand in [
        .timestamped(pass: pass, writes: .init(querySet: query)),
        .timestamped(pass: pass, writes: .init(querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 0)),
        .timestamped(pass: pass, writes: .init(querySet: query, endOfPassWriteIndex: 4)),
        .timestamped(pass: timed, writes: .init(querySet: query, beginningOfPassWriteIndex: 1)),
        .timestamped(pass: .read(buffer: result, byteCount: 4), writes: .init(querySet: query, endOfPassWriteIndex: 1)),
        .resolveQuerySet(querySet: query, firstQuery: 4, queryCount: 0, destination: result),
        .resolveQuerySet(querySet: query, firstQuery: 3, queryCount: 2, destination: result),
        .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 1, destination: result, destinationOffset: 8),
        .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 1, destination: result, destinationOffset: 512),
        .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 1, destination: incorrectUsage),
    ] {
        await #expect(throws: GPUError.self) { try await device.submitCommands([invalid]) }
    }
    #expect(await host.submits == 0)
    let ticket = try await device.submitCommands([
        timed,
        .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 4, destination: result, destinationOffset: 256),
    ])
    await host.started.wait()
    try await device.destroyQuerySet(query)
    #expect(await host.live.contains(query))
    await #expect(throws: GPUError.self) { try await device.submitCommands([timed]) }
    await host.complete.release()
    _ = try await ticket.wait()
    #expect(await host.live.isEmpty)
    await device.close()
}

#if canImport(Metal)
@Test func resolvesComputeAndRenderTimestampsAcrossSubmissionsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.timestampQuery])
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(timestampCompute), to: .msl))
    let source = ShaderInput.wgsl(timestampRender)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let query = try await device.createQuerySet(.init(type: .timestamp, count: 6))
    let result = try await device.createBuffer(.init(size: 304, usage: [.queryResolve, .copyDestination, .copySource]))
    let readback = try await device.createBuffer(.init(size: 304, usage: [.copyDestination, .mapRead]))
    let output = try await device.createBuffer(.init(size: 256, usage: .storage))
    let positions = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let capture: [BufferCommand] = [
        .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 6, destination: result, destinationOffset: 256),
        .copy(source: result, destination: readback, byteCount: 304),
        .read(buffer: readback, byteCount: 304),
    ]
    let initialize: [BufferCommand] = [
        .write(buffer: result, offset: 0, words: [0x12345678]),
        .write(buffer: result, offset: 256, words: Array(repeating: 0xffffffff, count: 12)),
        .write(buffer: positions, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
    ]
    let computePass = BufferCommand.dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1))
    let renderPass = BufferCommand.render(pipeline: render, target: target, clear: .transparentBlack,
                                          vertexCount: 3, vertexBuffers: [.init(buffer: positions)])
    let timestamps: [BufferCommand] = [
        .timestamped(pass: computePass, writes: .init(querySet: query, beginningOfPassWriteIndex: 1, endOfPassWriteIndex: 2)),
        .timestamped(pass: renderPass, writes: .init(querySet: query, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5)),
    ]
    let snapshots = try await device.submitCommands(initialize + capture + timestamps + capture).wait()
    let previous = try #require(snapshots.last)
    let repeated = try await device.submitCommands(capture).wait()
    #expect(repeated == [previous])
    #expect(snapshots.map { $0[0] } == [0x12345678, 0x12345678])
    #expect(snapshots[0][64..<76].allSatisfy { $0 == 0 })
    let samples = stride(from: 64, to: 76, by: 2).map { UInt64(previous[$0]) | (UInt64(previous[$0 + 1]) << 32) }
    #expect(samples[0] == 0 && samples[3] == 0)
    #expect(samples[1] > 0 && samples[2] >= samples[1])
    #expect(samples[4] > 0 && samples[5] >= samples[4])
    #expect(!samples.contains(UInt64.max))
    let emptyCompute = BufferCommand.dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output],
                                                      workgroups: .init(x: 0))
    let emptyRender = BufferCommand.render(pipeline: render, target: target, clear: .transparentBlack,
                                           vertexCount: 0, vertexBuffers: [.init(buffer: positions)])
    let emptyPasses: [BufferCommand] = [
        .timestamped(pass: emptyCompute, writes: .init(querySet: query, beginningOfPassWriteIndex: 0)),
        .timestamped(pass: emptyRender, writes: .init(querySet: query, endOfPassWriteIndex: 3)),
    ]
    let emptySnapshots = try await device.submitCommands(emptyPasses + capture).wait()
    let empty = try #require(emptySnapshots.first)
    let computeBoundary = UInt64(empty[64]) | (UInt64(empty[65]) << 32)
    let renderBoundary = UInt64(empty[70]) | (UInt64(empty[71]) << 32)
    #expect(computeBoundary > 0 && computeBoundary < UInt64.max)
    #expect(renderBoundary > 0 && renderBoundary < UInt64.max)
    #expect(empty[66..<70] == previous[66..<70])
    #expect(empty[72..<76] == previous[72..<76])
    let arguments = try await device.createBuffer(.init(size: 32, usage: [.indirect, .copyDestination]))
    let indirectCompute = BufferCommand.dispatchComputeIndirect(pipeline: compute,
        bindings: [.init(binding: 0): .init(buffer: output)], indirectBuffer: arguments)
    let indirectRender = BufferCommand.renderIndirect(pipeline: render, target: target, colorLoad: .clear(.transparentBlack),
        indirectBuffer: arguments, indirectOffset: 16, vertexBuffers: [.init(buffer: positions)])
    let indirectPasses: [BufferCommand] = [
        .write(buffer: arguments, offset: 0, words: [0, 1, 1, 0, 0, 1, 0, 0]),
        .timestamped(pass: indirectCompute, writes: .init(querySet: query, beginningOfPassWriteIndex: 0)),
        .timestamped(pass: indirectRender, writes: .init(querySet: query, endOfPassWriteIndex: 3)),
    ]
    let indirectSnapshots = try await device.submitCommands(indirectPasses + capture).wait()
    let indirect = try #require(indirectSnapshots.first)
    let indirectComputeBoundary = UInt64(indirect[64]) | (UInt64(indirect[65]) << 32)
    let indirectRenderBoundary = UInt64(indirect[70]) | (UInt64(indirect[71]) << 32)
    #expect(indirectComputeBoundary > computeBoundary && indirectComputeBoundary < UInt64.max)
    #expect(indirectRenderBoundary > renderBoundary && indirectRenderBoundary < UInt64.max)
    #expect(device.capabilities.timestampPeriodNanoseconds > 0)
    await device.close()
}
#endif
