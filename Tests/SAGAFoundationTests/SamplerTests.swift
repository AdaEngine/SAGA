import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let sampledTextureShader = """
    @group(2) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
    @group(2) @binding(5) var image: texture_2d<f32>;
    @group(2) @binding(6) var pointSampler: sampler;
    @group(2) @binding(7) var linearSampler: sampler;
    const east = vec2i(2, -1) + vec2i(-1, 1);
    fn blend(uv: vec2f, mip: f32) -> vec4f { return textureSampleLevel(image, linearSampler, uv, mip); }
    @compute @workgroup_size(1) fn run() {
        let uv = vec2f(0.4375f, 0.5f);
        out[0] = textureSampleLevel(image, pointSampler, uv, 0, east,);
        out[1] = blend(uv, 0.0);
        out[2] = textureSampleGrad(image, pointSampler, vec2f(-0.25f, 0.5f), vec2f(0.25f, 0.0f), vec2f(0.0f, 0.25f), vec2i(-1, 0));
        out[3] = textureLoad(image, vec2u(1u, 0u), 0u);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsExplicitTextureSamplingWithIndependentSamplerBindings(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(sampledTextureShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .msl:
        #expect(source.contains("saga_sampler_2 [[sampler(0)]]"))
        #expect(source.contains("saga_sampler_3 [[sampler(1)]]"))
        #expect(source.contains(".sample(") && source.contains(".read("))
    case .hlsl:
        #expect(source.contains("register(t5, space2)"))
        #expect(source.contains("register(s6, space2)") && source.contains("register(s7, space2)"))
        #expect(source.contains(".SampleLevel(") && source.contains(".SampleGrad("))
    case .glsl:
        #expect(source.contains("textureLod(saga_pair_1_3"))
        #expect(source.contains("textureGradOffset(saga_pair_1_2"))
        #expect(source.contains("textureLodOffset(saga_pair_1_2"))
        #expect(source.contains("texelFetch(saga_texture_1"))
        #expect(shader.glslTextureSamplerBindings.map(\.flatBindingIndex) == [1, 2])
        #expect(shader.glslTextureSamplerBindings.map(\.sampler) == [.init(group: 2, binding: 6), .init(group: 2, binding: 7)])
    case .wgsl:
        #expect(source.contains("@group(2) @binding(6)"))
        #expect(source.contains("textureSampleLevel("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected format")
    }
    #expect(shader.samplerBindings.map(\.binding) == [.init(group: 2, binding: 6), .init(group: 2, binding: 7)])
}

@Test(arguments: [
    "textureSampleLevel(image, image, vec2f(0), 0.0)",
    "textureSampleLevel(image, state, vec2u(0), 0.0)",
    "textureSampleLevel(image, state, vec2f(0), 0u)",
])
func rejectsInvalidExplicitTextureSamplingOperands(call: String) {
    let source = """
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var state: sampler;
        @compute @workgroup_size(1) fn run() { let color = \(call); }
        """
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func samplerFilteringAndAddressingUseBoundResourcesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(sampledTextureShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let point = try await device.createSampler(.init(addressModeU: .repeat))
        let linear = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
        let input = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
        let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                        usage: [.copyDestination, .textureBinding]))
        let out = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        func dispatch(_ point: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: pipeline, bindings: [.init(group: 2, binding: 0): out],
                             workgroups: .init(x: 1), textures: [.init(group: 2, binding: 5): image],
                             samplers: [.init(group: 2, binding: 6): point, .init(group: 2, binding: 7): linear])
        }
        await #expect(throws: GPUError.validation("Invalid sampler binding")) {
            try await device.submitCommands([dispatch(image)])
        }
        let command = dispatch(point)
        #expect(command.resources.contains(point))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0xff0000ff, 0xff00ff00]),
            .copyBufferToTexture(source: input, layout: .init(), destination: image, region: .init(width: 2, height: 1)),
            command,
            .copy(source: out, destination: read, byteCount: 64),
            .read(buffer: read, byteCount: 64),
        ])
        try await device.destroySampler(point)
        let snapshots = try await ticket.wait()
        let words = try #require(snapshots.first)
        let expected: [Float] = [0, 1, 0, 1, 0.625, 0.375, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        #expect(words.count == expected.count)
        for (actual, expected) in zip(words.map(Float.init(bitPattern:)), expected) {
            #expect(abs(actual - expected) < 0.004)
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
