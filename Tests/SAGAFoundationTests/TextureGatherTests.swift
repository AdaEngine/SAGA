import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let textureGatherShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
    @group(0) @binding(1) var image: texture_2d<f32>;
    @group(0) @binding(2) var state: sampler;
    const green = 2u - 1u;
    fn gatherRed(uv: vec2f) -> vec4f { return textureGather(0, image, state, uv); }
    @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
        if id.x == 0u {
            let uv = vec2f(0.33333334, 0.5);
            out[0] = gatherRed(uv);
            out[1] = textureGather(green, image, state, uv, vec2(1, 0));
            out[2] = textureGather(2i, image, state, uv, vec2i(-1, 0),);
            out[3] = textureGather(3u, image, state, uv);
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsTextureGatherWithConstantComponentAndOffset(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(textureGatherShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("textureGather(1u,"))
        #expect(source.contains("textureDimensions(") && source.contains("floor("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains(".gather("))
        #expect(source.contains("float2(int2(1, 0))"))
        #expect(source.contains("int2(0), component::y"))
        #expect(source.contains("int2(0), component::w"))
    case .hlsl:
        #expect(source.contains(".GatherRed(") && source.contains(".GatherGreen("))
        #expect(source.contains(".GatherBlue(") && source.contains(".GatherAlpha("))
        #expect(source.contains("saga_buffer_0[(3u) * 4u]"))
    case .glsl:
        #expect(source.contains("textureGather(saga_pair_1_2"))
        #expect(source.contains("textureGatherOffset(saga_pair_1_2"))
        #expect(source.contains("ivec2(1, 0), 1)"))
        #expect(shader.glslTextureSamplerBindings.count == 1)
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    ("-1i", ShaderDiagnosticCode.invalidType),
    ("4u", ShaderDiagnosticCode.invalidType),
    ("component", ShaderDiagnosticCode.invalidType),
    ("1.0", ShaderDiagnosticCode.typeMismatch),
])
func rejectsInvalidTextureGatherComponents(component: String, diagnostic: ShaderDiagnosticCode) throws {
    let source = """
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var state: sampler;
        @compute @workgroup_size(1) fn run() {
            let component = 1i;
            let texels = textureGather(\(component), image, state, vec2f(0.5));
        }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid gather component accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == diagnostic)
    }
}

#if canImport(Metal)
@Test func gathersFourTexelsInWGSLOrderFromTextureViewsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(textureGatherShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let sampler = try await device.createSampler(.init(addressModeU: .repeat))
        let image = try await device.createTexture(.init(width: 6, height: 4, format: .rgba8Unorm,
            usage: [.copyDestination, .textureBinding], mipLevelCount: 2))
        let view = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
        let input = try await device.createBuffer(.init(size: 268, usage: [.copyDestination, .copySource]))
        let out = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        func dispatch(_ texture: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 1): texture], samplers: [.init(binding: 2): sampler])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0x8c190a14, 0x962d1e28, 0xa041323c]),
            .write(buffer: input, offset: 256, words: [0xaa554650, 0xb4695a64, 0xbe7d6e78]),
            .copyBufferToTexture(source: input, layout: .init(bytesPerRow: 256), destination: image,
                                 region: .init(width: 3, height: 2, mipLevel: 1)),
            dispatch(view),
            .copy(source: out, destination: read, byteCount: 64),
            .read(buffer: read, byteCount: 64),
            dispatch(image),
            .copy(source: out, destination: read, byteCount: 64),
            .read(buffer: read, byteCount: 64),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        let lanes = snapshots[0].map { (Float(bitPattern: $0) * 255).rounded() }
        #expect(lanes == [80, 100, 40, 20, 90, 110, 50, 30, 125, 85, 25, 65, 170, 180, 150, 140])
        #expect(snapshots[1] == Array(repeating: 0, count: 16))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
