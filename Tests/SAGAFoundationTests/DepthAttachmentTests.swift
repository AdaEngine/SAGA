import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let depthShader = """
    struct VertexOut {
        @builtin(position) position: vec4f,
        @location(0) tint: vec4f,
    }
    @vertex fn vs(@location(0) point: vec3f, @location(1) tint: vec4f) -> VertexOut {
        return VertexOut(vec4f(point, 1.0), tint);
    }
    @fragment fn fs(@location(0) tint: vec4f) -> @location(0) vec4f { return tint; }
    """

#if canImport(Metal)
@Test func depthAttachmentRejectsOccludedFragmentsAndLoadsTheSelectedMipOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(depthShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let layouts = [VertexBufferLayout(stride: 28, attributes: [
            .init(location: 0, offset: 0, format: .float32x3),
            .init(location: 1, offset: 12, format: .float32x4),
        ])]
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
            depthState: .init(compare: .less, writeEnabled: true))
        let readOnly = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
            depthState: .init(compare: .always, writeEnabled: false))
        let target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.copySource, .renderAttachment]))
        let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
            usage: [.copySource, .renderAttachment], mipLevelCount: 2))
        let depthView = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
        let vertices = try await device.createBuffer(.init(size: 168, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let streams = [VertexBufferBinding(buffer: vertices)]
        func draw(_ count: UInt32, depth attachment: RenderDepthAttachment?) -> BufferCommand {
            .render(pipeline: pipeline, target: target, clear: blue, vertexCount: count,
                    vertexBuffers: streams, depth: attachment)
        }
        let load = RenderDepthAttachment(texture: depthView, loadAction: .load)
        let clear = RenderDepthAttachment(texture: depthView, loadAction: .clear(1))
        for attachment in [
            nil, RenderDepthAttachment(texture: target, loadAction: .clear(1)),
            .init(texture: depth, loadAction: .clear(1)),
            .init(texture: depthView, loadAction: .clear(.nan)),
            .init(texture: depthView, loadAction: .clear(1.5)),
        ] {
            await #expect(throws: GPUError.self) {
                try await device.submitCommands([draw(3, depth: attachment)])
            }
        }
        let triangles: [Float] = [
            -1, -1, 0.25, 0, 1, 0, 1,
             3, -1, 0.25, 0, 1, 0, 1,
            -1,  3, 0.25, 0, 1, 0, 1,
            -1, -1, 0.75, 1, 0, 0, 1,
             3, -1, 0.75, 1, 0, 0, 1,
            -1,  3, 0.75, 1, 0, 0, 1,
        ]
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: triangles.map(\.bitPattern)),
            .write(buffer: indices, offset: 0, words: [0x00040003, 5]),
            draw(3, depth: load),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
            draw(6, depth: clear),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
            .renderIndexed(pipeline: pipeline, target: target, clear: blue,
                           indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                           vertexBuffers: streams, depth: load),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
            .renderIndexed(pipeline: readOnly, target: target, clear: blue,
                           indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                           vertexBuffers: streams, depth: load),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
            .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 8),
        ])
        try await device.destroyTextureView(depthView)
        let expected: [[UInt32]] = [
            [0xffff0000, 0xffff0000], [0xff00ff00, 0xff00ff00],
            [0xffff0000, 0xffff0000], [0xff0000ff, 0xff0000ff],
            Array(repeating: Float(0.25).bitPattern, count: 2),
        ]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
