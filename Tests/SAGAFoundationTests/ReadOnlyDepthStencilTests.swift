import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let readOnlyDepthShader = """
    @group(0) @binding(0) var image: texture_depth_2d;
    fn position(i: u32, z: f32) -> vec4f {
        if i == 0u { return vec4f(-1, -1, z, 1); }
        if i == 1u { return vec4f(3, -1, z, 1); }
        return vec4f(-1, 3, z, 1);
    }
    @vertex fn seed(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.25); }
    @vertex fn near(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.125); }
    @vertex fn far(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.5); }
    @fragment fn clearColor() -> @location(0) vec4f { return vec4f(0); }
    @fragment fn sampleDepth() -> @location(0) vec4f {
        return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), textureLoad(image, vec2u(0), 0u) == 0.25);
    }
    """

@Test func rejectsWritesToReadOnlyDepthAndStencilAspects() async throws {
    let backend = RenderTestBackend()
    let device = try await Device.make(backend: backend)
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(readOnlyDepthShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "seed", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "clearColor", to: .msl)
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float, usage: .renderAttachment))
    let stencil = try await device.createTexture(.init(width: 1, height: 1, format: .depth24PlusStencil8, usage: .renderAttachment))
    let depthWriter = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, depthState: .init())
    await #expect(throws: GPUError.validation("Render pipeline writes to a read-only depth attachment")) {
        _ = try await device.submitCommands([
            .render(pipeline: depthWriter, target: color, clear: .transparentBlack, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .readOnly)),
        ])
    }
    let stencilState = RenderDepthState(format: .depth24PlusStencil8, writeEnabled: false,
        stencil: .init(front: .init(pass: .replace)))
    let stencilWriter = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, depthState: stencilState)
    let readOnlyStencil = RenderDepthAttachment(texture: stencil, loadAction: .clear(1), stencil: .init(loadAction: .readOnly))
    await #expect(throws: GPUError.validation("Render pipeline writes to a read-only stencil attachment")) {
        _ = try await device.submitCommands([
            .render(pipeline: stencilWriter, target: color, clear: .transparentBlack, vertexCount: 3, depth: readOnlyStencil),
        ])
    }
    let culled = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, depthState: stencilState, primitive: .init(cullMode: .front))
    await backend.release.release()
    _ = try await device.submitCommands([
        .render(pipeline: culled, target: color, clear: .transparentBlack, vertexCount: 3, depth: readOnlyStencil),
    ]).wait()
    await device.close()
}

#if canImport(Metal)
@Test func samplesReadOnlyDepthAttachmentWhileDepthTestingOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(readOnlyDepthShader)
        let writer = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "seed", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "clearColor", to: .msl),
            colorFormat: .rgba8Unorm, depthState: .init())
        let reader = try compiler.compileRenderStage(source, entryPoint: "sampleDepth", to: .msl)
        let near = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "near", to: .msl), fragment: reader,
            colorFormat: .rgba8Unorm, depthState: .init(writeEnabled: false))
        let far = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "far", to: .msl), fragment: reader,
            colorFormat: .rgba8Unorm, depthState: .init(writeEnabled: false))
        let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding, .copySource], mipLevelCount: 2))
        let attachment = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
        let sampled = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        func draw(_ pipeline: RenderPipeline, load: RenderDepthAttachment.LoadAction) -> BufferCommand {
            .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 0): sampled], depth: .init(texture: attachment, loadAction: load))
        }
        await #expect(throws: GPUError.validation("Render attachment cannot be bound as a sampled texture")) {
            _ = try await device.submitCommands([draw(near, load: .load)])
        }
        let ticket = try await device.submitCommands([
            .render(pipeline: writer, target: color, clear: .transparentBlack, vertexCount: 3,
                    depth: .init(texture: attachment)),
            draw(near, load: .readOnly),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
            draw(far, load: .readOnly),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
            .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 8),
        ])
        #expect(try await ticket.wait() == [[0xff00ff00, 0xff00ff00], [0, 0],
                                            Array(repeating: Float(0.25).bitPattern, count: 2)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
