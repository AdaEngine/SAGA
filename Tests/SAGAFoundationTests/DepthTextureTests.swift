import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let depthLoadShader = """
    @group(2) @binding(1) var image: texture_depth_2d;
    @group(0) @binding(0) var<storage, read_write> output: array<f32, 8>;
    fn fetch(p: vec2i) -> f32 { return textureLoad(image, p, 0i); }
    @compute @workgroup_size(1) fn run() {
        let size = textureDimensions(image, 0u);
        output[0u] = fetch(vec2i(0));
        output[1u] = fetch(vec2i(-1, 0));
        output[2u] = textureLoad(image, vec2u(0xffffffffu, 0u), 0u);
        output[3u] = textureLoad(image, vec2u(0), 1u);
        output[4u] = f32(size.x);
        output[5u] = f32(size.y);
        output[6u] = f32(textureNumLevels(image));
        output[7u] = textureLoad(image, vec2u(1, 0), 0u);
    }
    """

private let depthRenderShader = """
    @group(2) @binding(1) var image: texture_depth_2d;
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point, 0.25, 1.0);
    }
    @fragment fn writeDepth() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    @fragment fn readDepth() -> @location(0) vec4f {
        return vec4f(textureLoad(image, vec2u(0u), 0u) * 4.0, 0, 0, 1);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsScalarDepthTextureLoadsAndQueries(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(depthLoadShader), to: format)
    let resource = try #require(shader.sampledTextures.first)
    #expect(resource.binding == .init(group: 2, binding: 1) && resource.isDepth)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("texture_depth_2d"))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl:
        #expect(body.contains("depth2d<float, access::sample>"))
        #expect(body.contains(".read(") && body.contains(".get_num_mip_levels()"))
    case .hlsl:
        #expect(body.contains("Texture2D<float>") && body.contains(".Load("))
    case .glsl:
        #expect(body.contains("uniform sampler2D") && body.contains("texelFetch("))
    default: Issue.record("Unexpected format")
    }
}

#if canImport(Metal)
@Test func readsRasterizedDepthFromTextureAndViewsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(depthLoadShader), to: .msl))
        let source = ShaderInput.wgsl(depthRenderShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let writer = try compiler.compileRenderStage(source, entryPoint: "writeDepth", to: .msl)
        let reader = try compiler.compileRenderStage(source, entryPoint: "readDepth", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let render = try await device.createRenderPipeline(
            vertex: vertex, fragment: writer, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
            depthState: .init(compare: .less))
        let preview = try await device.createRenderPipeline(
            vertex: vertex, fragment: reader, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
        let conflicting = try await device.createRenderPipeline(
            vertex: vertex, fragment: reader, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
            depthState: .init(compare: .always, writeEnabled: false))
        let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding], mipLevelCount: 2))
        let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
        let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .textureBinding, .copySource]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let depthSlot = ComputeBindingSlot(group: 2, binding: 1)
        let streams = [VertexBufferBinding(buffer: vertices)]
        let depthAttachment = RenderDepthAttachment(texture: view, loadAction: .clear(1))
        func dispatch(_ texture: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output],
                             workgroups: .init(x: 1), textures: [depthSlot: texture])
        }
        await #expect(throws: GPUError.self) { try await device.submitCommands([dispatch(color)]) }
        await #expect(throws: GPUError.validation("Render attachment cannot be bound as a sampled texture")) {
            try await device.submitCommands([
                .render(pipeline: conflicting, target: color, clear: clear, vertexCount: 3,
                        vertexBuffers: streams, textures: [depthSlot: view], depth: depthAttachment),
            ])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            .render(pipeline: render, target: color, clear: clear, vertexCount: 3,
                    vertexBuffers: streams, depth: depthAttachment),
            .render(pipeline: preview, target: color, clear: clear, vertexCount: 3,
                    vertexBuffers: streams, textures: [depthSlot: view]),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
            dispatch(view),
            .copy(source: output, destination: read, byteCount: 32),
            .read(buffer: read, byteCount: 32),
            dispatch(depth),
            .copy(source: output, destination: read, byteCount: 32),
            .read(buffer: read, byteCount: 32),
        ])
        let viewed: [Float] = [0.25, 0, 0, 0, 2, 1, 1, 0.25]
        let whole: [Float] = [0, 0, 0, 0.25, 4, 2, 2, 0]
        let expected: [[UInt32]] = [[0xff0000ff, 0xff0000ff], viewed.map(\.bitPattern), whole.map(\.bitPattern)]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
