import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let topologyShader = """
    struct VertexOut {
        @builtin(position) position: vec4f,
        @location(0) marker: f32,
    }
    @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0, 1); }
    @vertex fn structured(@location(0) p: vec2f) -> VertexOut {
        return VertexOut(vec4f(p, 0, 1), 1.0);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    """

@Test func validatesStripIndexFormatBeforeSubmission() async throws {
    let backend = RenderTestBackend()
    let device = try await Device.make(backend: backend)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(topologyShader), entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl(topologyShader), entryPoint: "fs", to: .msl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    func pipeline(_ primitive: RenderPrimitiveState) async throws -> RenderPipeline {
        try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                              vertexBuffers: layouts, primitive: primitive)
    }
    await #expect(throws: GPUError.validation("A strip index format requires a strip topology")) {
        _ = try await pipeline(.init(topology: .triangleList, stripIndexFormat: .uint16))
    }
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let vertices = try await device.createBuffer(.init(size: 24, usage: .vertex))
    let indices = try await device.createBuffer(.init(size: 12, usage: .index))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    for primitive: RenderPrimitiveState in [.init(topology: .lineStrip), .init(topology: .triangleStrip, stripIndexFormat: .uint32)] {
        let strip = try await pipeline(primitive)
        await #expect(throws: GPUError.validation("Indexed strips require a matching strip index format")) {
            _ = try await device.submitCommands([
                .renderIndexed(pipeline: strip, target: target, clear: clear, indexBuffer: .init(buffer: indices, format: .uint16),
                               draw: .init(indexCount: 3), vertexBuffers: [.init(buffer: vertices)]),
            ])
        }
    }
    #expect(await backend.submitted == 0)
    await device.close()
}

#if canImport(Metal)
@Test(arguments: [false, true])
func drawsPointsLinesAndStripsWithPrimitiveRestartOnMetal(structured: Bool) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(.wgsl(topologyShader), entryPoint: structured ? "structured" : "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(.wgsl(topologyShader), entryPoint: "fs", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let states: [RenderPrimitiveState] = [
            .init(topology: .pointList), .init(topology: .lineList), .init(topology: .lineStrip),
            .init(topology: .triangleStrip), .init(topology: .triangleStrip, stripIndexFormat: .uint16),
            .init(topology: .lineStrip, stripIndexFormat: .uint32),
        ]
        var pipelines: [RenderPipeline] = []
        for primitive in states {
            pipelines.append(try await device.createRenderPipeline(
                vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts, primitive: primitive))
        }
        let target = try await device.createTexture(.init(width: 8, height: 4, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let vertices = try await device.createBuffer(.init(size: 176, usage: [.vertex, .copyDestination]))
        let short = try await device.createBuffer(.init(size: 20, usage: [.index, .copyDestination]))
        let long = try await device.createBuffer(.init(size: 20, usage: [.index, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 800, usage: [.copyDestination, .mapRead]))
        let positions: [(Float, Float)] = [
            (0.5, 0.5), (7.5, 3.5),
            (1.5, 0.5), (6.5, 0.5), (6.5, 3.5), (1.5, 3.5),
            (1, 0), (1, 4), (7, 0), (7, 4),
            (0, 0), (0, 4), (2, 0), (2, 4), (6, 0), (6, 4), (8, 0), (8, 4),
            (0.5, 1.5), (2.5, 1.5), (5.5, 1.5), (7.5, 1.5),
        ]
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        var commands: [BufferCommand] = [
            .write(buffer: vertices, offset: 0, words: positions.flatMap { x, y in
                [(x / 4 - 1).bitPattern, (1 - y / 2).bitPattern]
            }),
            .write(buffer: short, offset: 0, words: [0x00010009, 0x00030002, 0xffff0004, 0x00060005, 0x00080007]),
            .write(buffer: long, offset: 0, words: [0, 1, .max, 2, 3]),
        ]
        let draws: [BufferCommand] = [
            .render(pipeline: pipelines[0], target: target, clear: clear, vertexCount: 2, vertexBuffers: [.init(buffer: vertices)]),
            .render(pipeline: pipelines[1], target: target, clear: clear, vertexCount: 4, vertexBuffers: [.init(buffer: vertices)], firstVertex: 2),
            .render(pipeline: pipelines[2], target: target, clear: clear, vertexCount: 4, vertexBuffers: [.init(buffer: vertices)], firstVertex: 2),
            .render(pipeline: pipelines[3], target: target, clear: clear, vertexCount: 4, vertexBuffers: [.init(buffer: vertices)], firstVertex: 6),
            .renderIndexed(pipeline: pipelines[4], target: target, clear: clear, indexBuffer: .init(buffer: short, format: .uint16),
                           draw: .init(indexCount: 9, firstIndex: 1, baseVertex: -1), vertexBuffers: [.init(buffer: vertices, offset: 80)]),
            .renderIndexed(pipeline: pipelines[5], target: target, clear: clear, indexBuffer: .init(buffer: long, format: .uint32),
                           draw: .init(indexCount: 5), vertexBuffers: [.init(buffer: vertices, offset: 144)]),
        ]
        for draw in draws {
            commands += [draw, .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
                         .read(buffer: read, byteCount: 800)]
        }
        let ticket = try await device.submitCommands(commands)
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 6 && snapshots.allSatisfy { $0.count == 200 })
        let green: UInt32 = 0xff00ff00
        let black: UInt32 = 0xff000000
        let pointPixels = (0..<4).flatMap { row in snapshots[0][(row * 64)..<(row * 64 + 8)] }
        #expect(pointPixels.filter { $0 == green }.count == 2)
        #expect([snapshots[0][0], snapshots[0][199], snapshots[0][68]] == [green, green, black])
        #expect([snapshots[1][3], snapshots[1][70], snapshots[1][195]] == [green, black, green])
        #expect([snapshots[2][3], snapshots[2][70], snapshots[2][195]] == [green, green, green])
        #expect([snapshots[3][3], snapshots[3][195], snapshots[3][64]] == [green, green, black])
        #expect([snapshots[4][64], snapshots[4][68], snapshots[4][71]] == [green, black, green])
        #expect([snapshots[5][65], snapshots[5][68], snapshots[5][70]] == [green, black, green])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
