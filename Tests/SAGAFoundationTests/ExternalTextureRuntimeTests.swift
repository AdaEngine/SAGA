import Testing
import SAGATypes
import SAGABackend
import SAGACore
import SAGACompiler
import SAGAWebGPU
import SAGAShaderIR

private actor ExternalTextureHost: BrowserGPUComputeHost, ExternalTextureBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    let started = Gate()
    let submitGate: Gate?
    var textures: [ResourceID: ExternalTextureDescriptor] = [:]
    var expired: Set<UInt64> = []

    init(submitGate: Gate? = nil) { self.submitGate = submitGate }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) {}
    func destroyBuffer(id: ResourceID) {}
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) {}
    func destroyComputePipeline(id: ResourceID) {}
    func allocateExternalTexture(id: ResourceID, descriptor: ExternalTextureDescriptor) { textures[id] = descriptor }
    func destroyExternalTexture(id: ResourceID) { textures.removeValue(forKey: id) }
    func expire(_ token: UInt64) { expired.insert(token) }
    func submit(_ submission: Submission) {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        await started.release()
        if let submitGate { await submitGate.wait() }
        for id in submission.resources {
            if let texture = textures[id], expired.contains(texture.videoFrame) {
                throw GPUError.validation("VideoFrame has expired")
            }
        }
        return []
    }
}

private let externalTextureRuntimeShader = """
    @group(0) @binding(0) var video: texture_external;
    @compute @workgroup_size(1) fn run() {
        _ = textureLoad(video, vec2u(0));
    }
    """

@Test func externalTextureRuntimeChecksKindAndRetiresAfterQueuedUse() async throws {
    let gate = Gate()
    let host = ExternalTextureHost(submitGate: gate)
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    let shader = try IndependentShaderCompiler().compileWebGPUCompute(.wgsl(externalTextureRuntimeShader))
    let pipeline = try await device.createComputePipeline(shader)
    let descriptor = ExternalTextureDescriptor(videoFrame: 7, colorSpace: .displayP3)
    let texture = try await device.importExternalTexture(descriptor)
    #expect(await host.textures[texture] == descriptor)
    let buffer = try await device.createBuffer(.init(size: 16, usage: .storage))
    await #expect(throws: GPUError.validation("Invalid external texture binding")) {
        try await device.submitCommands([.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                                                           textures: [.init(binding: 0): buffer])])
    }
    let ordinary = try IndependentShaderCompiler().compileWebGPUCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @compute @workgroup_size(1) fn run() { _ = textureLoad(image, vec2u(0), 0); }
        """))
    let ordinaryPipeline = try await device.createComputePipeline(ordinary)
    await #expect(throws: GPUError.validation("Invalid sampled texture binding")) {
        try await device.submitCommands([.dispatchCompute(pipeline: ordinaryPipeline, bindings: [:], workgroups: .init(x: 1),
                                                           textures: [.init(binding: 0): texture])])
    }
    let command = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                                                textures: [.init(binding: 0): texture])
    let ticket = try await device.submitCommands([command])
    await host.started.wait()
    try await device.destroyExternalTexture(texture)
    #expect(await host.textures[texture] != nil)
    await gate.release()
    _ = try await ticket.wait()
    #expect(await host.textures[texture] == nil)
    await #expect(throws: GPUError.self) { try await device.submitCommands([command]) }
    await device.close()
}

@Test func externalTextureRuntimePropagatesFrameExpiry() async throws {
    let host = ExternalTextureHost()
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    let shader = try IndependentShaderCompiler().compileWebGPUCompute(.wgsl(externalTextureRuntimeShader))
    let pipeline = try await device.createComputePipeline(shader)
    let expired = try await device.importExternalTexture(.init(videoFrame: 8))
    let fresh = try await device.importExternalTexture(.init(videoFrame: 9))
    await host.expire(8)
    func dispatch(_ texture: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1), textures: [.init(binding: 0): texture])
    }
    let failed = try await device.submitCommands([dispatch(expired)])
    await #expect(throws: GPUError.validation("VideoFrame has expired")) { try await failed.wait() }
    _ = try await device.submitCommands([dispatch(fresh)]).wait()
    try await device.destroyExternalTexture(expired)
    try await device.destroyExternalTexture(fresh)
    #expect(await host.textures.isEmpty)
    await device.close()
}
