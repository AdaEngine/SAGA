#if canImport(Metal)
import SAGACore
import SAGACompiler
@_spi(IndexedDrawGrouping) import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

@Test
func groupedMetalIndexedDrawsMatchSeparatePasses() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl("""
            @vertex fn vertexMain(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0, 1); }
            @fragment fn red() -> @location(0) vec4f { return vec4f(1, 0, 0, 0.5); }
            @fragment fn green() -> @location(0) vec4f { return vec4f(0, 1, 0, 0.5); }
            @fragment fn white() -> @location(0) vec4f { return vec4f(1, 1, 1, 1); }
            """)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vertexMain", to: .msl)
        let layouts: [VertexBufferLayout] = [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let blend = RenderColorState(blend: .init(color: .init(sourceFactor: .sourceAlpha, destinationFactor: .oneMinusSourceAlpha)))
        func pipeline(_ entryPoint: String) async throws -> RenderPipeline {
            let fragment = try compiler.compileRenderStage(source, entryPoint: entryPoint, to: .msl)
            return try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                          vertexBuffers: layouts, colorState: blend)
        }
        let red = try await pipeline("red")
        let green = try await pipeline("green")
        let white = try await pipeline("white")
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let short = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let word = try await device.createBuffer(.init(size: 24, usage: [.index, .copyDestination]))
        let positions: [Float] = [-1, -1, 3, -1, -1, 3]
        _ = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: positions.map(\.bitPattern)),
            .write(buffer: short, offset: 0, words: [0x0001_0000, 2]),
            .write(buffer: word, offset: 0, words: [0, 1, 2, 0, 1, 99]),
        ]).wait()

        func compare(separate: Bool) async throws -> [[UInt32]] {
            let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                usage: [.renderAttachment, .copySource]))
            let secondTarget = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                usage: [.renderAttachment, .copySource]))
            let secondView = try await device.createTextureView(texture: secondTarget, .init())
            let result = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
            let clipped = RenderRasterState(
                viewport: .init(x: 0, y: 0, width: 4, height: 8), scissor: .init(x: 0, y: 0, width: 4, height: 8))
            let vertexBuffers: [VertexBufferBinding] = [.init(buffer: vertices)]
            let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
            let commands: [BufferCommand] = [
                .renderIndexed(pipeline: red, target: target, clear: blue,
                    indexBuffer: .init(buffer: short, format: .uint16), draw: .init(indexCount: 3),
                    vertexBuffers: vertexBuffers, raster: clipped),
                .renderIndexed(pipeline: green, target: target, colorLoad: .load,
                    indexBuffer: .init(buffer: word, format: .uint32), draw: .init(indexCount: 3), vertexBuffers: vertexBuffers),
                .renderIndexed(pipeline: white, target: target, colorLoad: .load,
                    indexBuffer: .init(buffer: word, format: .uint32), draw: .init(indexCount: 3, firstIndex: 3),
                    vertexBuffers: vertexBuffers),
                .renderIndexed(pipeline: white, target: secondView, clear: .transparentBlack,
                    indexBuffer: .init(buffer: short, format: .uint16), draw: .init(indexCount: 3),
                    vertexBuffers: vertexBuffers),
                .renderIndexed(pipeline: green, target: target, colorLoad: .load,
                    indexBuffer: .init(buffer: word, format: .uint32), draw: .init(indexCount: 3),
                    vertexBuffers: vertexBuffers),
                .copyTextureToBuffer(source: target, destination: result, bytesPerRow: 256),
                .read(buffer: result, byteCount: 1824),
                .copyTextureToBuffer(source: secondTarget, destination: result, bytesPerRow: 256),
                .read(buffer: result, byteCount: 1824),
                .renderIndexed(pipeline: green, target: target, clear: .transparentBlack,
                    indexBuffer: .init(buffer: short, format: .uint16), draw: .init(indexCount: 3), vertexBuffers: vertexBuffers),
                .renderIndexed(pipeline: red, target: target, colorLoad: .load,
                    indexBuffer: .init(buffer: word, format: .uint32), draw: .init(indexCount: 3),
                    vertexBuffers: vertexBuffers, raster: clipped),
                .copyTextureToBuffer(source: target, destination: result, bytesPerRow: 256),
                .read(buffer: result, byteCount: 1824),
            ]
            if !separate {
                let groups = Dictionary(uniqueKeysWithValues: commands.indices.compactMap { index in
                    if case .renderIndexed = commands[index] { return (index, MetalIndexedDrawGrouping.color) }
                    return nil
                })
                return try await MetalIndexedDrawGrouping.$commands.withValue(groups) {
                    try await device.submitCommands(commands).wait()
                }
            }
            var output: [[UInt32]] = []
            for command in commands { output += try await device.submitCommands([command]).wait() }
            return output
        }
        let reference = try await compare(separate: true)
        let grouped = try await compare(separate: false)
        try #require(grouped.count == 3)
        #expect(grouped == reference)
        #expect(grouped[0].first != grouped[0].last, "The group must honor the clipped red pass.")
        #expect(grouped[0].last != 0xffffffff, "The bad index must cancel only its own draw.")
        #expect(grouped[0].first != grouped[1].first, "A copy must preserve the separately drawn texture view.")
        #expect(grouped.contains { $0.first == 0xffffffff }, "The incompatible texture-view attachment must render.")
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
