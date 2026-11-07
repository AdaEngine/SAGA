import Testing
import SAGATypes
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGACompiler
import SAGAWebGPU
@testable import SAGAMetal

private let program = """
    @group(0) @binding(1) var<storage, read> source: array<u32, 4>;
    @group(0) @binding(3) var<storage, read_write> destination: array<u32, 4>;

    @compute @workgroup_size(2, 2)
    fn run(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x + id.y * 2u;
        destination[i] = source[i] * 3u + 7u;
    }
    """

private let sourceSlot = ComputeBindingSlot(binding: 1)
private let destinationSlot = ComputeBindingSlot(binding: 3)

private actor ComputeTestBackend: BrowserGPUComputeHost {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    let allocationStarted = Gate()
    let allocationGate: Gate?
    var pipelines: Set<ResourceID> = []
    var submissions = 0
    var lastShader: ComputeShaderArtifact?

    init(allocationGate: Gate? = nil) { self.allocationGate = allocationGate }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) {}
    func destroyBuffer(id: ResourceID) {}
    func submit(_ submission: Submission) {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) -> [[UInt32]] {
        submissions += 1
        return []
    }
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) async throws {
        pipelines.insert(pipeline.id)
        lastShader = shader
        await allocationStarted.release()
        if let allocationGate { await allocationGate.wait() }
    }
    func destroyComputePipeline(id: ResourceID) { pipelines.remove(id) }
    func close(owner: UInt64) { pipelines = pipelines.filter { $0.device != owner } }
}

@Test func browserQueueUsesValidatedWGSLAndExplicitComputeHost() async throws {
    let host = ComputeTestBackend()
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(program), to: .wgsl)
    guard case .text(.wgsl, let source) = shader.artifact else {
        Issue.record("Expected emitted WGSL")
        return
    }
    #expect(source != program)
    #expect(source.contains("@compute @workgroup_size(2, 2, 1)"))
    #expect(source.contains("@group(0) @binding(1) var<storage, read>"))
    #expect(source.contains("@group(0) @binding(3) var<storage, read_write>"))
    let parsed = try compiler.parseAndValidate(.wgsl(source))
    #expect(parsed.module.entryPoints.first?.name == shader.entryPoint)
    #expect(parsed.module.entryPoints.first?.workgroupSize == shader.workgroupSize)
    #expect(parsed.module.globalVariables.values.compactMap(\.binding) == shader.storageBindings.map(\.binding))

    let msl = try compiler.compileCompute(.wgsl(program), to: .msl)
    await #expect(throws: GPUError.validation("Browser requires a compiled WGSL compute shader")) {
        try await device.createComputePipeline(msl)
    }
    #expect(await host.lastShader == nil)
    let pipeline = try await device.createComputePipeline(shader)
    #expect(await host.lastShader == shader)
    let input = try await device.createBuffer(.init(size: 16, usage: .storage))
    let destination = try await device.createBuffer(.init(size: 16, usage: .storage))
    let tooSmall = try await device.createBuffer(.init(size: 12, usage: .storage))
    await #expect(throws: GPUError.validation("Command has invalid resource, usage, size or alignment")) {
        try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [sourceSlot: input, destinationSlot: tooSmall],
                             workgroups: .init(x: 1))
        ])
    }
    #expect(await host.submissions == 0)
    let bindings = [sourceSlot: input, destinationSlot: destination]
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1))
    ])
    #expect(try await ticket.wait().isEmpty)
    #expect(await host.submissions == 1)
    await device.close()
    #expect(await host.pipelines.isEmpty)
}

private actor LegacyBrowserHost: BrowserGPUHost {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    var resourceUses = 0
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) {}
    func destroyBuffer(id: ResourceID) {}
    func submit(_ submission: Submission) { resourceUses += 1 }
}

@Test func browserWithoutComputeHostRejectsCommandsWithoutFallback() async throws {
    let host = LegacyBrowserHost()
    let device = try await Device.make(backend: BrowserBackend(host: host))
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(program), to: .wgsl)
    await #expect(throws: GPUError.unsupported("Backend does not implement compute pipelines")) {
        try await device.createComputePipeline(shader)
    }
    await #expect(throws: GPUError.unsupported("Backend does not implement command submission")) {
        try await device.submitCommands([])
    }
    #expect(await host.resourceUses == 0)
    let resource = try await device.createBuffer(.init(size: 16, usage: .storage))
    try await device.submitResourceUse([resource]).wait()
    #expect(await host.resourceUses == 1)
    await device.close()
}

@Test func computeDispatchRejectsInvalidBindingsBeforeSubmission() async throws {
    let backend = ComputeTestBackend()
    let device = try await Device.make(backend: backend)
    let other = try await Device.make(backend: backend)
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(program), to: .msl)
    let pipeline = try await device.createComputePipeline(shader)
    let foreignPipeline = try await other.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 16, usage: .storage))
    let output = try await device.createBuffer(.init(size: 16, usage: .storage))
    let tooSmall = try await device.createBuffer(.init(size: 12, usage: .storage))
    let wrongUsage = try await device.createBuffer(.init(size: 16, usage: .copySource))
    let foreign = try await other.createBuffer(.init(size: 16, usage: .storage))
    let scope = try await device.pushErrorScope(.validation)
    let bindings = [sourceSlot: input, destinationSlot: output]
    let invalid: [BufferCommand] = [
        .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1)),
        .dispatchCompute(pipeline: pipeline, bindings: [sourceSlot: input, destinationSlot: tooSmall], workgroups: .init(x: 1)),
        .dispatchCompute(pipeline: pipeline, bindings: [sourceSlot: input, destinationSlot: wrongUsage], workgroups: .init(x: 1)),
        .dispatchCompute(pipeline: pipeline, bindings: [sourceSlot: input, destinationSlot: input], workgroups: .init(x: 1)),
        .dispatchCompute(pipeline: pipeline, bindings: [sourceSlot: input, destinationSlot: foreign], workgroups: .init(x: 1)),
        .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 65_536)),
        .dispatchCompute(pipeline: foreignPipeline, bindings: bindings, workgroups: .init(x: 1)),
    ]
    for command in invalid {
        await #expect(throws: GPUError.self) { try await device.submitCommands([command]) }
    }
    #expect(await backend.submissions == 0)
    #expect(try await device.popErrorScope(scope) != nil)
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1))
    ])
    #expect(ticket.sequence == 1)
    #expect(try await ticket.wait().isEmpty)
    await device.close()
    #expect(await backend.pipelines == [foreignPipeline.id])
    await other.close()
    #expect(await backend.pipelines.isEmpty)
}

@Test func cancelledPipelineAllocationRollsBackItsBackendHandle() async throws {
    let gate = Gate()
    let backend = ComputeTestBackend(allocationGate: gate)
    let device = try await Device.make(backend: backend)
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(program), to: .msl)
    let task = Task { try await device.createComputePipeline(shader) }
    await backend.allocationStarted.wait()
    task.cancel()
    await gate.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await backend.pipelines.isEmpty)
    await device.close()
}

#if canImport(Metal)
@Test func wgslPipelineRunsThroughDeviceQueueAndMetalReadback() async throws {
    let backend = try MetalBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    let compiler = IndependentShaderCompiler()
    do {
        let shader = try compiler.compileCompute(.wgsl(program), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let bindings = [sourceSlot: input, destinationSlot: output]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0, 1, 2, .max]),
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 0)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1, y: 2)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ])
        try await device.destroyBuffer(input)
        #expect(try await ticket.wait() == [[0, 0, 0, 0], [7, 10, 13, 4]])
        let emptyShader = try compiler.compileCompute(.wgsl("@compute @workgroup_size(1) fn run() {}"), to: .msl)
        let empty = try await device.createComputePipeline(emptyShader)
        let noBuffers = try await device.submitCommands([
            .dispatchCompute(pipeline: empty, bindings: [:], workgroups: .init(x: 1))
        ])
        #expect(try await noBuffers.wait().isEmpty)
        let hlsl = try compiler.compileCompute(.wgsl(program), to: .hlsl)
        await #expect(throws: GPUError.self) { try await device.createComputePipeline(hlsl) }
        #expect(await backend.resourceCountsForTesting().pipelines == 2)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
    #expect(await backend.resourceCountsForTesting().pipelines == 0)
}
#endif
