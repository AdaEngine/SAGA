import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let integerTextureDraw = """
    @vertex fn vs(@location(0) xy: vec2f) -> @builtin(position) vec4f { return vec4f(xy, 0, 1); }
    @fragment fn signedColor() -> @location(0) vec4i { return vec4i(-17, 23, -3, 1); }
    @fragment fn unsignedColor() -> @location(0) vec4u { return vec4u(249, 1, 200, 7); }
    """

private let integerTextureRead = """
    @group(0) @binding(0) var signedImage: texture_multisampled_2d<i32>;
    @group(0) @binding(1) var unsignedImage: texture_multisampled_2d<u32>;
    @group(0) @binding(2) var single: texture_2d<u32>;
    @group(0) @binding(3) var<storage, read_write> signedOut: array<vec4i, 5>;
    @group(0) @binding(4) var<storage, read_write> unsignedOut: array<vec4u, 7>;
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        signedOut[lane] = textureLoad(signedImage, vec2i(0), i32(lane));
        unsignedOut[lane] = textureLoad(unsignedImage, vec2u(0), lane);
        if lane == 0u {
            signedOut[4] = textureLoad(signedImage, vec2i(0), -1i);
            unsignedOut[4] = textureLoad(unsignedImage, vec2u(0), 0xffffffffu);
            unsignedOut[5] = textureLoad(single, vec2u(0), 0u);
            unsignedOut[6] = vec4u(textureNumSamples(signedImage), textureNumSamples(unsignedImage),
                                  textureDimensions(unsignedImage));
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsIntegerTextureLoadsAndFragmentOutputs(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let compute = try compiler.compileCompute(.wgsl(integerTextureRead), to: format)
    let signed = try compiler.compileRenderStage(.wgsl(integerTextureDraw), entryPoint: "signedColor", to: format)
    let unsigned = try compiler.compileRenderStage(.wgsl(integerTextureDraw), entryPoint: "unsignedColor", to: format)
    #expect(compute.sampledTextures.map(\.sampleKind) == [.sint, .uint, .uint])
    #expect(compute.sampledTextures.map(\.isMultisampled) == [true, true, false])
    #expect(signed.colorOutputKind == .sint && unsigned.colorOutputKind == .uint)
    guard case .text(_, let loads) = compute.artifact, case .text(_, let sint) = signed.artifact,
          case .text(_, let uint) = unsigned.artifact else { Issue.record("Expected text artifacts"); return }
    switch format {
    case .wgsl:
        #expect(loads.contains("texture_multisampled_2d<i32>") && loads.contains("texture_multisampled_2d<u32>"))
        #expect(loads.contains("texture_2d<u32>"))
        for source in [loads, sint, uint] { _ = try compiler.parseAndValidate(.wgsl(source)) }
    case .msl:
        #expect(loads.contains("texture2d_ms<int, access::read>") && loads.contains("texture2d_ms<uint, access::read>"))
        #expect(sint.contains("int4 color [[color(0)]]") && uint.contains("uint4 color [[color(0)]]"))
    case .hlsl:
        #expect(loads.contains("Texture2DMS<int4>") && loads.contains("Texture2DMS<uint4>"))
        #expect(sint.contains("int4 saga_fragment(") && uint.contains("uint4 saga_fragment("))
    case .glsl:
        #expect(loads.contains("isampler2DMS") && loads.contains("usampler2DMS") && loads.contains("uniform usampler2D "))
        #expect(sint.contains("out ivec4 saga_color;") && uint.contains("out uvec4 saga_color;"))
    default: Issue.record("Unexpected target")
    }
}

@Test func integerTexturesRejectFloatSampling() throws {
    let source = """
        @group(0) @binding(0) var image: texture_2d<i32>;
        @group(0) @binding(1) var state: sampler;
        @compute @workgroup_size(1) fn run() { let value = textureSampleLevel(image, state, vec2f(0.5), 0.0); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Integer texture accepted float sampling")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func rendersAndLoadsSignedAndUnsignedTextureSamplesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let draw = ShaderInput.wgsl(integerTextureDraw)
        let vertex = try compiler.compileRenderStage(draw, entryPoint: "vs", to: .msl)
        let signed = try compiler.compileRenderStage(draw, entryPoint: "signedColor", to: .msl)
        let unsigned = try compiler.compileRenderStage(draw, entryPoint: "unsignedColor", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let signedPipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: signed, colorFormat: .rgba8Sint, vertexBuffers: layouts, sampleCount: 4)
        let unsignedPipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: unsigned, colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4)
        let singlePipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: unsigned, colorFormat: .rgba8Uint, vertexBuffers: layouts)
        await #expect(throws: GPUError.self) {
            _ = try await device.createRenderPipeline(
                vertex: vertex, fragment: signed, colorFormat: .rgba8Uint, vertexBuffers: layouts)
        }
        let reader = try await device.createComputePipeline(compiler.compileCompute(.wgsl(integerTextureRead), to: .msl))
        let usage: TextureUsage = [.renderAttachment, .textureBinding]
        let a = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Sint, usage: usage, sampleCount: 4))
        let b = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: usage, sampleCount: 4))
        let c = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: usage))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let signedOut = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let unsignedOut = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        func render(_ pipeline: RenderPipeline, to target: ResourceID) -> BufferCommand {
            .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)])
        }
        func load(_ signedTexture: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: reader, bindings: [.init(binding: 3): signedOut, .init(binding: 4): unsignedOut],
                             workgroups: .init(x: 1),
                             textures: [.init(binding: 0): signedTexture, .init(binding: 1): b, .init(binding: 2): c])
        }
        await #expect(throws: GPUError.validation("Invalid sampled texture binding")) {
            _ = try await device.submitCommands([load(b)])
        }
        await #expect(throws: GPUError.self) {
            _ = try await device.submitCommands([
                .render(pipeline: unsignedPipeline, target: b, clear: .init(red: -1, green: 0, blue: 0, alpha: 0),
                        vertexCount: 0, vertexBuffers: [.init(buffer: vertices)]),
            ])
        }
        await #expect(throws: GPUError.validation("Invalid color resolve attachment")) {
            _ = try await device.submitCommands([
                .render(pipeline: unsignedPipeline, target: b, clear: clear, vertexCount: 3,
                        vertexBuffers: [.init(buffer: vertices)], resolve: .init(texture: c)),
            ])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            render(signedPipeline, to: a), render(unsignedPipeline, to: b), render(singlePipeline, to: c),
            load(a),
            .copy(source: signedOut, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
            .copy(source: unsignedOut, destination: read, byteCount: 112), .read(buffer: read, byteCount: 112),
        ])
        let signedWords: [UInt32] = Array(repeating: [Int32(-17), 23, -3, 1].map(UInt32.init(bitPattern:)), count: 4)
            .flatMap { $0 } + [0, 0, 0, 0]
        let unsignedSample: [UInt32] = [249, 1, 200, 7]
        let unsignedWords: [UInt32] = Array(repeating: unsignedSample, count: 4).flatMap { $0 } +
            [0, 0, 0, 0] + unsignedSample + [4, 4, 1, 1]
        #expect(try await ticket.wait() == [signedWords, unsignedWords])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

private let integerGatherShader = """
    @group(0) @binding(0) var signedImage: texture_2d<i32>;
    @group(0) @binding(1) var unsignedImage: texture_2d<u32>;
    @group(0) @binding(2) var state: sampler;
    @group(0) @binding(3) var<storage, read_write> signedOut: array<vec4i, 2>;
    @group(0) @binding(4) var<storage, read_write> unsignedOut: array<vec4u, 2>;
    const alpha = 1u + 2u;
    fn gatherSigned(uv: vec2f) -> vec4i { return textureGather(alpha, signedImage, state, uv); }
    @compute @workgroup_size(1) fn run() {
        let uv = vec2f(0.33333334, 0.5);
        signedOut[0] = gatherSigned(uv);
        signedOut[1] = textureGather(alpha, signedImage, state, uv, vec2i(1, 0));
        unsignedOut[0] = textureGather(2i, unsignedImage, state, uv, vec2i(-1, 0));
        unsignedOut[1] = textureGather(alpha, unsignedImage, state, uv);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsSignedAndUnsignedTextureGather(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(integerGatherShader), to: format)
    #expect(shader.sampledTextures.map(\.sampleKind) == [.sint, .uint])
    #expect(shader.samplerBindings.first?.requiresNonFiltering == true)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("textureGather(3u,"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains("texture2d<int, access::sample>"))
        #expect(source.contains("texture2d<uint, access::sample>"))
        #expect(source.contains(".gather(") && source.contains("component::w"))
    case .hlsl:
        #expect(source.contains("Texture2D<int4>") && source.contains("Texture2D<uint4>"))
        #expect(source.contains(".GatherAlpha(") && source.contains(".GatherBlue("))
    case .glsl:
        #expect(source.contains("uniform isampler2D saga_pair_0_2"))
        #expect(source.contains("uniform usampler2D saga_pair_1_2"))
        #expect(source.contains("textureGatherOffset(saga_pair_1_2"))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    ("texture_multisampled_2d<i32>", "sampler"),
    ("texture_2d<i32>", "sampler_comparison"),
])
func rejectsInvalidIntegerGatherOperands(texture: String, sampler: String) throws {
    let source = """
        @group(0) @binding(0) var image: \(texture);
        @group(0) @binding(1) var state: \(sampler);
        @compute @workgroup_size(1) fn run() { let value = textureGather(0, image, state, vec2f(0.5)); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid integer gather accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func gathersIntegerChannelsAndOffsetsFromMipViewsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compute = try await device.createComputePipeline(
            IndependentShaderCompiler().compileCompute(.wgsl(integerGatherShader), to: .msl))
        let signed = try await device.createTexture(.init(width: 6, height: 4, format: .rgba8Sint,
            usage: [.copyDestination, .textureBinding], mipLevelCount: 2))
        let unsigned = try await device.createTexture(.init(width: 6, height: 4, format: .rgba8Uint,
            usage: [.copyDestination, .textureBinding], mipLevelCount: 2))
        let signedView = try await device.createTextureView(texture: signed, .init(baseMipLevel: 1))
        let unsignedView = try await device.createTextureView(texture: unsigned, .init(baseMipLevel: 1))
        let nearest = try await device.createSampler(.init(addressModeU: .repeat))
        let linear = try await device.createSampler(.init(addressModeU: .repeat, mipmapFilter: .linear))
        let input = try await device.createBuffer(.init(size: 268, usage: [.copyDestination, .copySource]))
        let signedOut = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let unsignedOut = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        func dispatch(_ sampler: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 3): signedOut, .init(binding: 4): unsignedOut],
                             workgroups: .init(x: 1), textures: [.init(binding: 0): signedView, .init(binding: 1): unsignedView],
                             samplers: [.init(binding: 2): sampler])
        }
        await #expect(throws: GPUError.validation("Invalid sampler binding")) {
            _ = try await device.submitCommands([dispatch(linear)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0x8c190a14, 0x962d1e28, 0xa041323c]),
            .write(buffer: input, offset: 256, words: [0xaa554650, 0xb4695a64, 0xbe7d6e78]),
            .copyBufferToTexture(source: input, layout: .init(bytesPerRow: 256), destination: signed,
                                 region: .init(width: 3, height: 2, mipLevel: 1)),
            .copyBufferToTexture(source: input, layout: .init(bytesPerRow: 256), destination: unsigned,
                                 region: .init(width: 3, height: 2, mipLevel: 1)),
            dispatch(nearest),
            .copy(source: signedOut, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
            .copy(source: unsignedOut, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        ])
        let signedLanes: [Int32] = [-86, -76, -106, -116, -76, -66, -96, -106]
        #expect(try await ticket.wait() == [signedLanes.map(UInt32.init(bitPattern:)),
                                            [125, 85, 25, 65, 170, 180, 150, 140]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
