#if canImport(Metal)
import SAGACore
import SAGACompiler
@_spi(IndexedDrawGrouping) import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

@Test(arguments: [TextureFormat.depth32FloatStencil8, .depth24PlusStencil8])
func groupedLocalShadowDepthMatchesSeparatePasses(format: TextureFormat) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let shader = ShaderInput.wgsl("""
            @vertex fn vertexMain(@location(0) p: vec3f) -> @builtin(position) vec4f { return vec4f(p, 1); }
            @fragment fn fragmentMain(@builtin(position) p: vec4f) -> @location(0) f32 { return p.z; }
            """)
        let vertex = try compiler.compileRenderStage(shader, entryPoint: "vertexMain", to: .msl)
        let fragment = try compiler.compileRenderStage(shader, entryPoint: "fragmentMain", to: .msl)
        let layout = VertexBufferLayout(stride: 12, attributes: [.init(location: 0, offset: 0, format: .float32x3)])
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .r32Float, vertexBuffers: [layout],
            depthState: .init(format: format)
        )
        let vertices = try await device.createBuffer(.init(size: 108, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
        let triangles: [Float] = [
            -1, -1, 0.25, 3, -1, 0.25, -1, 3, 0.25,
            -1, -1, 0.75, 3, -1, 0.75, -1, 3, 0.75,
            -1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5,
        ]
        _ = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: triangles.map(\.bitPattern)),
            .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        ]).wait()

        func render(separate: Bool) async throws -> [[UInt32]] {
            let target = try await device.createTexture(.init(width: 8, height: 8, format: .r32Float,
                usage: [.renderAttachment, .copySource]))
            let depth = try await device.createTexture(.init(width: 8, height: 8,
                format: format, usage: [.renderAttachment, .copySource]))
            let copy = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
            let white = RenderClearColor(red: 1, green: 1, blue: 1, alpha: 1)
            func draw(base: Int32, load: RenderColorLoadAction = .load,
                      depthLoad: RenderDepthAttachment.LoadAction = .load,
                      stencil: RenderStencilAttachmentState.LoadAction = .load,
                      raster: RenderRasterState = .init()) -> BufferCommand {
                .renderIndexed(pipeline: pipeline, target: target, colorLoad: load,
                    indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3, baseVertex: base),
                    vertexBuffers: [.init(buffer: vertices)],
                    depth: .init(texture: depth, loadAction: depthLoad, stencil: .init(loadAction: stencil)), raster: raster)
            }
            var commands: [BufferCommand] = [
                draw(base: 0, load: .clear(white), depthLoad: .clear(1), stencil: .clear(7),
                     raster: .init(scissor: .init(x: 0, y: 0, width: 4, height: 8))),
                draw(base: 3),
                .copyTextureToBuffer(source: target, destination: copy, bytesPerRow: 256),
                .read(buffer: copy, byteCount: 1824),
                draw(base: 0),
                draw(base: 3, depthLoad: .clear(0.625), stencil: .clear(19)),
                draw(base: 6),
                draw(base: 3, stencil: .clear(23)),
                draw(base: 6),
                .copyTextureToBuffer(source: target, destination: copy, bytesPerRow: 256),
                .read(buffer: copy, byteCount: 1824),
            ]
            if format == .depth32FloatStencil8 {
                commands.append(.copyTextureToBuffer(source: depth, destination: copy, bytesPerRow: 256, aspect: .depthOnly))
                commands.append(.read(buffer: copy, byteCount: 1824))
            }
            commands.append(.copyTextureToBuffer(source: depth, destination: copy, bytesPerRow: 256, aspect: .stencilOnly))
            commands.append(.read(buffer: copy, byteCount: 1824))
            if !separate {
                let groups = Dictionary(uniqueKeysWithValues: commands.indices.compactMap { index in
                    if case .renderIndexed = commands[index] {
                        return (index, MetalIndexedDrawGrouping([.color, .depthStencil]))
                    }
                    return nil
                })
                return try await MetalIndexedDrawGrouping.$commands.withValue(groups) {
                    try await device.submitCommands(commands).wait()
                }
            }
            var reads: [[UInt32]] = []
            for command in commands {
                reads += try await device.submitCommands([command]).wait()
            }
            return reads
        }
        let reference = try await render(separate: true)
        let grouped = try await render(separate: false)
        try #require(grouped.count == (format == .depth32FloatStencil8 ? 4 : 3))
        #expect(grouped == reference)
        #expect(grouped[0].first == Float(0.25).bitPattern)
        #expect(grouped[0].last == Float(0.75).bitPattern)
        #expect(grouped[1].first == Float(0.5).bitPattern)
        if format == .depth32FloatStencil8 { #expect(grouped[2].last == Float(0.5).bitPattern) }
        #expect(grouped.last?.first == 0x17171717, "A stencil-only clear must end the previous group.")
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
