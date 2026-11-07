import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let viewportShader = """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0.25, 1); }
        if i == 1u { return vec4f(3, -1, 0.25, 1); }
        return vec4f(-1, 3, 0.25, 1);
    }
    @fragment fn color(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
        return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), pixel.z == 0.625);
    }
    @fragment fn depth() -> @builtin(frag_depth) f32 { return 0.1; }
    """

@Test func validatesViewportAndScissorAgainstDeviceAndAttachment() throws {
    let target = ResourceID(device: 1, serial: 1)
    let pipeline = RenderPipeline(id: .init(device: 1, serial: 2), colorFormat: .rgba8Unorm)
    let texture = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: .renderAttachment)
    func validate(_ raster: RenderRasterState) throws {
        try BufferCommand.render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3, raster: raster)
            .validate(device: 1, descriptors: [:], textures: [target: texture])
    }
    try validate(.init(viewport: .init(x: -1, y: -1, width: 5, height: 5),
                       scissor: .init(x: 4, y: 4, width: 0, height: 0)))
    for viewport in [
        RenderViewport(x: .nan, y: 0, width: 1, height: 1),
        .init(x: 0, y: 0, width: -1, height: 1),
        .init(x: 0, y: 0, width: 1, height: 1, minDepth: 0.8, maxDepth: 0.2),
        .init(x: 0, y: 0, width: 1, height: 1, maxDepth: 1.1),
    ] {
        #expect(throws: GPUError.validation("Invalid render viewport")) { try validate(.init(viewport: viewport)) }
    }
    #expect(throws: GPUError.validation("Scissor rectangle exceeds the render attachment")) {
        try validate(.init(scissor: .init(x: .max, y: 0, width: 2, height: 1)))
    }
}

#if canImport(Metal)
@Test func appliesViewportScissorAndFragmentDepthRangeOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(viewportShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let color = try await device.createRenderPipeline(vertex: vertex,
            fragment: compiler.compileRenderStage(source, entryPoint: "color", to: .msl), colorFormat: .rgba8Unorm)
        let depth = try await device.createRenderPipeline(vertex: vertex,
            fragment: compiler.compileRenderStage(source, entryPoint: "depth", to: .msl), colorFormat: .rgba8Unorm,
            depthState: .init(compare: .less))
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource], mipLevelCount: 2))
        let view = try await device.createTextureView(texture: target, .init(baseMipLevel: 1))
        let depthTarget = try await device.createTexture(.init(width: 4, height: 4, format: .depth32Float,
            usage: [.renderAttachment, .copySource]))
        let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
        let raster = RenderRasterState(viewport: .init(x: 1, y: 1, width: 3, height: 2, minDepth: 0.5),
                                      scissor: .init(x: 0, y: 2, width: 3, height: 2))
        let ticket = try await device.submitCommands([
            .write(buffer: indices, offset: 0, words: [0, 1, 2]),
            .render(pipeline: color, target: view, clear: .transparentBlack, vertexCount: 3, raster: raster),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 784),
            .renderIndexed(pipeline: depth, target: view, colorLoad: .load, indexBuffer: .init(buffer: indices, format: .uint32),
                           draw: .init(indexCount: 3), depth: .init(texture: depthTarget, loadAction: .clear(0.75)), raster: raster),
            .copyTextureToBuffer(source: depthTarget, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 784),
            .render(pipeline: color, target: view, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3,
                    raster: .init(scissor: .init(x: 4, y: 4, width: 0, height: 0))),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 784),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 3 && snapshots.allSatisfy { $0.count == 196 })
        for y in 0..<4 {
            for x in 0..<4 {
                let inside = y == 2 && (x == 1 || x == 2)
                #expect(snapshots[0][y * 64 + x] == (inside ? 0xff00ff00 : 0))
                #expect(snapshots[1][y * 64 + x] == (inside ? Float(0.5) : 0.75).bitPattern)
                #expect(snapshots[2][y * 64 + x] == 0xffff0000)
            }
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
