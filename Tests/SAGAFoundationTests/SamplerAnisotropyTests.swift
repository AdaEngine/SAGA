import Testing
import SAGATypes
import SAGACore
import SAGAMetal
import SAGACompiler

@Test func samplerAnisotropyValidation() throws {
    #expect(SamplerDescriptor().maxAnisotropy == 1)
    try SamplerDescriptor(maxAnisotropy: 1).validate()
    for value: UInt16 in [2, 16, 17, .max] {
        let descriptor = SamplerDescriptor(minFilter: .linear, magFilter: .linear,
                                           mipmapFilter: .linear, maxAnisotropy: value)
        #expect(descriptor.maxAnisotropy == value)
        try descriptor.validate()
    }
    #expect(throws: GPUError.self) { try SamplerDescriptor(maxAnisotropy: 0).validate() }
    for descriptor in [
        SamplerDescriptor(minFilter: .nearest, magFilter: .linear, mipmapFilter: .linear, maxAnisotropy: 2),
        SamplerDescriptor(minFilter: .linear, magFilter: .nearest, mipmapFilter: .linear, maxAnisotropy: 2),
        SamplerDescriptor(minFilter: .linear, magFilter: .linear, mipmapFilter: .nearest, maxAnisotropy: 2),
    ] {
        #expect(throws: GPUError.self) { try descriptor.validate() }
    }
}

#if canImport(Metal)
@Test func anisotropicSamplerRunsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        await #expect(throws: GPUError.self) {
            try await device.createSampler(.init(maxAnisotropy: 0))
        }
        let sampler = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear,
                                                           mipmapFilter: .linear, maxAnisotropy: .max))
        let source = """
            @group(0) @binding(0) var image: texture_2d<f32>;
            @group(0) @binding(1) var state: sampler;
            @group(0) @binding(2) var<storage, read_write> output: array<vec4f, 1>;
            @compute @workgroup_size(1) fn run() {
                output[0] = textureSampleLevel(image, state, vec2f(0.5), 0.0);
            }
            """
        let pipeline = try await device.createComputePipeline(
            IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl))
        let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.copyDestination, .textureBinding]))
        let upload = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff0000ff]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(width: 1, height: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(group: 0, binding: 2): output],
                workgroups: .init(x: 1), textures: [.init(group: 0, binding: 0): image],
                samplers: [.init(group: 0, binding: 1): sampler]),
            .copy(source: output, destination: read, byteCount: 16),
            .read(buffer: read, byteCount: 16),
        ])
        let words = try #require(try await ticket.wait().first)
        #expect(words.map(Float.init(bitPattern:)) == [1, 0, 0, 1])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
