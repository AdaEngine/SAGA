import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let arraySamplingShader = """
    @group(0) @binding(0) var color: texture_2d_array<f32>;
    @group(0) @binding(1) var depth: texture_depth_2d_array;
    @group(0) @binding(2) var nearest: sampler;
    @group(0) @binding(3) var comparison: sampler_comparison;
    @group(0) @binding(4) var<storage, read_write> out: array<vec4f, 6>;
    @compute @workgroup_size(1) fn run() {
        let uv = vec2f(0.5);
        out[0] = textureSampleLevel(color, nearest, uv, -5i, 0.0);
        out[1] = textureSampleLevel(color, nearest, uv, 0xffffffffu, 0.0);
        out[2] = textureGather(0u, color, nearest, uv, -1i, vec2i(1, 0));
        let less = textureSampleCompareLevel(depth, comparison, uv, -1i, 0.5, vec2i(0));
        let greater = textureSampleCompareLevel(depth, comparison, uv, 99u, 0.5);
        let raw = textureSampleLevel(depth, nearest, uv, 0u, 0u);
        out[3] = vec4f(less, greater, raw, textureGather(depth, nearest, uv, 99u).x);
        out[4] = textureGatherCompare(depth, comparison, uv, 99u, 0.5);
        out[5] = textureSampleGrad(color, nearest, uv, 1u, vec2f(0), vec2f(0));
    }
    """

private let arrayDepthWriter = """
    @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0); }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsArraySamplingGatherAndComparison(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(arraySamplingShader), to: format)
    #expect(shader.sampledTextures.map(\.isArrayed) == [true, true])
    #expect(shader.samplerBindings.first?.requiresNonFiltering == true)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("textureSampleCompareLevel(") && body.contains("textureGatherCompare("))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl:
        #expect(body.contains("depth2d_array<float, access::sample>"))
        #expect(body.contains(".get_array_size()") && body.contains(".gather_compare("))
    case .hlsl:
        #expect(body.contains("Texture2DArray<float>") && body.contains(".SampleCmpLevelZero("))
        #expect(body.contains(".GatherCmp(") && body.contains("float3("))
    case .glsl:
        #expect(body.contains("uniform sampler2DArrayShadow"))
        #expect(body.contains("textureGradOffset(") && body.contains("vec2(0.0)"))
        #expect(body.contains("textureGather("))
    default: Issue.record("Unexpected format")
    }
}

#if canImport(Metal)
@Test func samplesAndGathersFromClampedArrayViewLayersOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(arrayDepthWriter)
        let render = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm, depthState: .init())
        let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(arraySamplingShader), to: .msl))
        let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
            usage: [.copyDestination, .textureBinding], arrayLayerCount: 3))
        let depth = try await device.createTexture(.init(width: 2, height: 2, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding], arrayLayerCount: 3))
        let colorView = try await device.createTextureView(texture: color, .init(baseArrayLayer: 1, arrayLayerCount: 2))
        let depthView = try await device.createTextureView(texture: depth, .init(baseArrayLayer: 1, arrayLayerCount: 2))
        let firstDepth = try await device.createTextureView(texture: depth, .init(dimension: .d2, baseArrayLayer: 1))
        let lastDepth = try await device.createTextureView(texture: depth, .init(dimension: .d2, baseArrayLayer: 2))
        let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
        let nearest = try await device.createSampler(.init(addressModeU: .repeat))
        let compare = try await device.createSampler(.init(compare: .less))
        let upload = try await device.createBuffer(.init(size: 776, usage: [.copySource, .copyDestination]))
        let out = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 96, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff0000ff]),
            .write(buffer: upload, offset: 256, words: [0xff0000ff, 0xff0000ff]),
            .write(buffer: upload, offset: 512, words: [0xff00ff00, 0xff00ff00]),
            .write(buffer: upload, offset: 768, words: [0xff00ff00, 0xff00ff00]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: color,
                                 region: .init(width: 2, height: 2, arrayLayer: 1)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 512, bytesPerRow: 256), destination: color,
                                 region: .init(width: 2, height: 2, arrayLayer: 2)),
            .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                    depth: .init(texture: firstDepth, loadAction: .clear(0.25))),
            .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                    depth: .init(texture: lastDepth, loadAction: .clear(0.75))),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 4): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): colorView, .init(binding: 1): depthView],
                             samplers: [.init(binding: 2): nearest, .init(binding: 3): compare]),
            .copy(source: out, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
        ])
        let expected: [Float] = [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0.25, 0.75, 1, 1, 1, 1, 0, 1, 0, 1]
        #expect(try await ticket.wait() == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

private let implicitArraySamplingShader = """
    @group(0) @binding(0) var color: texture_2d_array<f32>;
    @group(0) @binding(1) var depth: texture_depth_2d_array;
    @group(0) @binding(2) var linearFilter: sampler;
    @group(0) @binding(3) var nearest: sampler;
    @group(0) @binding(4) var comparison: sampler_comparison;
    @fragment fn fs() -> @location(0) vec4f {
        let uv = vec2f(0.5);
        let colorSample = textureSample(color, linearFilter, uv, 0u);
        let biased = textureSampleBias(color, linearFilter, uv, 1i, 0.5, vec2i(1, 0));
        let raw = textureSample(depth, nearest, uv, 0u);
        let compare = textureSampleCompare(depth, comparison, uv, 1i, 0.5);
        return vec4f(colorSample.r, biased.g, raw, compare);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsImplicitArraySamplingAndKeepsColorSamplerFilterable(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(implicitArraySamplingShader), entryPoint: "fs", to: format)
    #expect(shader.samplerBindings.map(\.requiresNonFiltering) == [false, true, false])
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl: _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl: #expect(body.contains(".sample(") && body.contains("bias(") && body.contains(".sample_compare("))
    case .hlsl: #expect(body.contains(".Sample(") && body.contains(".SampleBias(") && body.contains(".SampleCmp("))
    case .glsl: #expect(body.contains("textureOffset(") && body.contains("vec4("))
    default: Issue.record("Unexpected format")
    }
}

@Test func arrayLayerContributesToSamplingUniformity() throws {
    let source = """
        @group(0) @binding(0) var image: texture_2d_array<f32>;
        @group(0) @binding(1) var state: sampler;
        @fragment fn fs(@location(0) @interpolate(flat) layer: u32) -> @location(0) vec4f {
            let uv = vec2f(0.5);
            let selected = textureSampleLevel(image, state, uv, layer, 0.0);
            if selected.r > 0.5 { return textureSample(image, state, uv, 0u); }
            return vec4f(0);
        }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Implicit sampling accepted after branching on a varying layer")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .nonUniformControlFlow)
    }
}
