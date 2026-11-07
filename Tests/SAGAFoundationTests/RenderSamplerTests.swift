import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let renderSamplerShader = """
    @group(2) @binding(5) var image: texture_2d<f32>;
    @group(1) @binding(4) var linearSampler: sampler;
    @group(2) @binding(7) var pointSampler: sampler;

    fn point(uv: vec2f) -> vec4f { return textureSampleLevel(image, pointSampler, uv, 0); }

    @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f {
        let offset = point(vec2f(0.25, 0.5)).r;
        return vec4f(position.x + offset, position.y, 0.0, 1.0);
    }

    @fragment fn fs() -> @location(0) vec4f {
        let uv = vec2f(0.4375, 0.5);
        const east = vec2(1, 0);
        let filtered = textureSample(image, linearSampler, vec2f(-0.0625, 0.5), east);
        let biased = textureSampleBias(image, linearSampler, uv, 2.0, east);
        let grad = textureSampleGrad(image, pointSampler, vec2f(0.75, 0.5), vec2f(0.25, 0.0), vec2f(0.0, 0.25));
        return vec4f(point(uv).r * grad.g, filtered.g, biased.g, 1.0);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsRenderSamplersForReachableStages(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(renderSamplerShader), entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(.wgsl(renderSamplerShader), entryPoint: "fs", to: format)
    let shared = IRResourceBinding(group: 2, binding: 7)
    let linear = IRResourceBinding(group: 1, binding: 4)
    #expect(vertex.samplerBindings.map(\.binding) == [shared])
    #expect(fragment.samplerBindings.map(\.binding) == [linear, shared])
    guard case .text(_, let vs) = vertex.artifact, case .text(_, let fs) = fragment.artifact else {
        Issue.record("Expected source")
        return
    }
    switch format {
    case .msl:
        #expect(vs.contains("saga_sampler_2 [[sampler(0)]]"))
        #expect(fs.contains("saga_sampler_2 [[sampler(1)]]"))
        #expect(vs.contains(".sample(") && fs.contains(".sample("))
        #expect(fs.contains("bias(clamp(") && fs.contains("gradient2d("))
    case .hlsl:
        #expect(vs.contains("register(s7, space2)") && fs.contains("register(s7, space2)"))
        #expect(fs.contains("register(s4, space1)") && fs.contains(".SampleLevel("))
        #expect(fs.contains(".Sample(") && fs.contains(".SampleBias(") && fs.contains(".SampleGrad("))
    case .glsl:
        let pair = try #require(vertex.glslTextureSamplerBindings.first)
        #expect(pair.texture == .init(group: 2, binding: 5) && pair.sampler == shared)
        #expect(vertex.glslTextureSamplerBindings.count == 1)
        #expect(fragment.glslTextureSamplerBindings.map(\.sampler) == [linear, shared])
        #expect(fragment.glslTextureSamplerBindings.last == pair)
        #expect(pair.flatBindingIndex == 2)
        #expect(vs.contains("layout(binding=2) uniform sampler2D saga_pair_0_2"))
        #expect(fs.contains("layout(binding=2) uniform sampler2D saga_pair_0_2"))
        #expect(fs.contains("textureOffset(saga_pair_0_1"))
        #expect(fs.contains("ivec2(1, 0), clamp("))
        #expect(fs.contains("textureGrad(saga_pair_0_2"))
    case .wgsl:
        #expect(vs.contains("@group(2) @binding(7)"))
        #expect(fs.contains("@group(1) @binding(4)"))
        _ = try compiler.parseAndValidate(.wgsl(vs))
        _ = try compiler.parseAndValidate(.wgsl(fs))
    default: Issue.record("Unexpected output")
    }
}

#if canImport(Metal)
@Test func drawsWithSharedSamplersAndStageSpecificFilteringOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(.wgsl(renderSamplerShader), entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(.wgsl(renderSamplerShader), entryPoint: "fs", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
        let conflicting = try compiler.compileRenderStage(.wgsl("""
            @group(2) @binding(7) var image: texture_2d<f32>;
            @fragment fn fs() -> @location(0) vec4f { return textureLoad(image, vec2u(0), 0); }
            """), entryPoint: "fs", to: .msl)
        await #expect(throws: GPUError.validation("Render stages disagree on the resource type of a binding")) {
            try await device.createRenderPipeline(vertex: vertex, fragment: conflicting,
                                                  colorFormat: .rgba8Unorm, vertexBuffers: layouts)
        }
        let upload = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                        usage: [.copyDestination, .textureBinding]))
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let point = try await device.createSampler()
        let linear = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
        let textures: [ComputeBindingSlot: ResourceID] = [.init(group: 2, binding: 5): image]
        let sharedSlot = ComputeBindingSlot(group: 2, binding: 7)
        let linearSlot = ComputeBindingSlot(group: 1, binding: 4)
        let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let streams = [VertexBufferBinding(buffer: vertices)]
        func draw(_ filtered: ResourceID) -> BufferCommand {
            .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: streams,
                    textures: textures, samplers: [sharedSlot: point, linearSlot: filtered])
        }
        await #expect(throws: GPUError.validation("Invalid sampler binding")) {
            try await device.submitCommands([draw(image)])
        }
        let command = draw(linear)
        #expect(command.resources.contains(point) && command.resources.contains(linear))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ff00]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(width: 2, height: 1)),
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
            command,
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                           indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                           vertexBuffers: streams, textures: textures, samplers: [sharedSlot: point, linearSlot: point]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        try await device.destroySampler(linear)
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2 && snapshots.allSatisfy { $0.count == 456 })
        let pixels = snapshots.flatMap { [$0[4 * 64 + 1], $0[4 * 64 + 6]] }
        #expect(pixels == [0xffff0000, 0xffff60ff, 0xffff0000, 0xffff00ff])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
