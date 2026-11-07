import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let stencilShader = """
    @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0.5, 1); }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    """

@Test func validatesStencilFormatAndAttachmentAspects() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(stencilShader), entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl(stencilShader), entryPoint: "fs", to: .msl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    await #expect(throws: GPUError.validation("Stencil state requires a stencil attachment format")) {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            vertexBuffers: layouts, depthState: .init(stencil: .init(front: .init(compare: .equal))))
    }
    await #expect(throws: GPUError.validation("Invalid format for a storage texture")) {
        _ = try await device.createTexture(.init(width: 1, height: 1, format: .depth24PlusStencil8,
                                                 usage: [.renderAttachment, .storageBinding]))
    }
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: layouts, depthState: .init(format: .depth24PlusStencil8))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth24PlusStencil8, usage: .renderAttachment))
    let vertices = try await device.createBuffer(.init(size: 24, usage: .vertex))
    await #expect(throws: GPUError.validation("Stencil load action does not match the depth attachment format")) {
        _ = try await device.submitCommands([
            .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth)),
        ])
    }
    await device.close()
}

#if canImport(Metal)
@Test(arguments: [UInt32(1), 4])
func appliesStencilFrontBackAndFailureOperationsOnMetal(sampleCount: UInt32) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(stencilShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        func pipeline(_ compare: CompareFunction, _ stencil: RenderStencilState) async throws -> RenderPipeline {
            try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                vertexBuffers: layouts, depthState: .init(format: .depth24PlusStencil8, compare: compare,
                    writeEnabled: false, stencil: stencil), sampleCount: sampleCount)
        }
        let writer = try await pipeline(.less, .init(front: .init(pass: .replace), back: .init(pass: .invert), writeMask: 0x0f))
        let equal = RenderStencilFaceState(compare: .equal)
        let reader = try await pipeline(.always, .init(front: equal, back: equal, readMask: 0x0f, writeMask: 0))
        let failure = try await pipeline(.never, .init(
            front: .init(compare: .equal, fail: .incrementWrap, depthFail: .decrementWrap), readMask: 0x0f, writeMask: 0x0f))
        let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let target: ResourceID
        let resolve: RenderResolveAttachment?
        if sampleCount > 1 {
            target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                usage: .renderAttachment, sampleCount: sampleCount))
            resolve = .init(texture: color)
        } else {
            target = color
            resolve = nil
        }
        let baseMip: UInt32 = sampleCount == 1 ? 1 : 0
        let depth = try await device.createTexture(.init(width: 2 << baseMip, height: 1 << baseMip, format: .depth24PlusStencil8,
            usage: .renderAttachment, mipLevelCount: baseMip + 1, sampleCount: sampleCount))
        let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: baseMip))
        let vertices = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        func draw(_ pipeline: RenderPipeline, reference: UInt32, first: UInt32 = 0,
                  stencil: RenderStencilAttachmentState.LoadAction = .load,
                  depthLoad: RenderDepthAttachment.LoadAction = .load, x: UInt32? = nil) -> BufferCommand {
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)], firstVertex: first,
                    depth: .init(texture: view, loadAction: depthLoad,
                                 stencil: .init(loadAction: stencil, reference: reference)), resolve: resolve,
                    raster: .init(scissor: x.map { .init(x: $0, y: 0, width: 1, height: 1) }))
        }
        let copy = BufferCommand.copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256)
        let snapshot = BufferCommand.read(buffer: read, byteCount: 8)
        let positions: [Float] = [-1, -1, 3, -1, -1, 3, -1, -1, -1, 3, 3, -1]
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: positions.map(\.bitPattern)),
            draw(reader, reference: 0, stencil: .readOnly, depthLoad: .clear(0.75)), copy, snapshot,
            draw(writer, reference: 5, stencil: .clear(0x1a0), depthLoad: .readOnly, x: 0),
            draw(writer, reference: 0, first: 3, depthLoad: .readOnly, x: 1),
            draw(reader, reference: 5, stencil: .readOnly), copy, snapshot,
            draw(reader, reference: 15, stencil: .readOnly), copy, snapshot,
            draw(failure, reference: 5),
            draw(reader, reference: 4, stencil: .readOnly), copy, snapshot,
            draw(reader, reference: 0, stencil: .readOnly), copy, snapshot,
        ])
        let green: UInt32 = 0xff00ff00
        #expect(try await ticket.wait() == [[green, green], [green, 0], [0, green], [green, 0], [0, green]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
