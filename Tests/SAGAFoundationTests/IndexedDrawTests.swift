import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private func makeIndexedPipeline(_ device: Device) async throws -> RenderPipeline {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "v", to: .msl)
    let fragment = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "f", to: .msl)
    return try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                 vertexBuffers: vertexBufferLayouts)
}

@Test func indexedDrawRejectsInvalidIndexBindingsBeforeSubmission() async throws {
    let backend = RenderTestBackend()
    let device = try await Device.make(backend: backend)
    let pipeline = try await makeIndexedPipeline(device)
    let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .renderAttachment))
    let positions = try await device.createBuffer(.init(size: 36, usage: .vertex))
    let colors = try await device.createBuffer(.init(size: 48, usage: .vertex))
    let index = try await device.createBuffer(.init(size: 16, usage: .index))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let vertices: [VertexBufferBinding] = [.init(buffer: positions), .init(buffer: colors)]
    let count = IndexedDraw(indexCount: 3)
    for (binding, draw) in [
        (IndexBufferBinding(buffer: index, format: .uint16, offset: 1), count),
        (.init(buffer: index, format: .uint32, offset: 2), count),
        (.init(buffer: index, format: .uint16, size: 4), count),
        (.init(buffer: index, format: .uint16, offset: 8, size: 10), count),
        (.init(buffer: positions, format: .uint16), count),
        (.init(buffer: index, format: .uint32), .init(indexCount: 3, firstIndex: 2)),
        (.init(buffer: index, format: .uint32), .init(indexCount: .max, firstIndex: .max)),
    ] {
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([
                .renderIndexed(pipeline: pipeline, target: texture, clear: clear, indexBuffer: binding,
                               draw: draw, vertexBuffers: vertices),
            ])
        }
    }
    // A trailing byte does not create another uint16 index, but the first three fit.
    try BufferCommand.renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                                    indexBuffer: .init(buffer: index, format: .uint16, size: 7),
                                    draw: count, vertexBuffers: vertices)
        .validate(device: device.identity, descriptors: [
            positions: .init(size: 36, usage: .vertex),
            colors: .init(size: 48, usage: .vertex),
            index: .init(size: 16, usage: .index),
        ], textures: [texture: .init(width: 8, height: 8, format: .rgba8Unorm, usage: .renderAttachment)])
    #expect(await backend.submitted == 0)
    await device.close()
}

#if canImport(Metal)
@Test func indexedProceduralDrawUsesVertexIndexWithoutVertexBuffersOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(.wgsl(triangleProgram), entryPoint: "vertexMain", to: .msl)
        let fragment = try compiler.compileRenderStage(.wgsl(triangleProgram), entryPoint: "fragmentMain", to: .msl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let index = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let ticket = try await device.submitCommands([
            .write(buffer: index, offset: 0, words: [1, 2, 3]),
            .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                           indexBuffer: .init(buffer: index, format: .uint32, offset: 12, size: 0),
                           draw: .init(indexCount: 0)),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                           indexBuffer: .init(buffer: index, format: .uint32),
                           draw: .init(indexCount: 3, baseVertex: -1)),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(snapshots[0].first == 0xffff0000)
        #expect(snapshots[1].first == 0xff00ff00)
        #expect(snapshots[1].last == 0xffff0000)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(arguments: [IndexFormat.uint16, .uint32])
func indexedDrawUsesIndexOrderAndSuppressesOutOfRangeFetchesOnMetal(format: IndexFormat) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await makeIndexedPipeline(device)
        let positions = try await device.createBuffer(.init(size: 36, usage: [.vertex, .copyDestination]))
        let colors = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 32, usage: [.index, .copyDestination]))
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        // The first and last indices are sentinels. The selected indices need baseVertex = -1.
        let words: [UInt32] = format == .uint16 ? [0x0001ffff, 0x00020003, 0x00030001, 0xffffffff] :
            [.max, 0, 1, 3, 2, 1, 3, .max]
        let first: UInt32 = format == .uint16 ? 0 : 1
        let binding = IndexBufferBinding(buffer: indices, format: format, offset: UInt64(format.byteSize))
        let vertices: [VertexBufferBinding] = [.init(buffer: positions), .init(buffer: colors)]
        let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let xy: [Float] = [-1, 1, 0, -1, -1, 0, 1, 1, 0]
        let rgb: [Float] = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        let ticket = try await device.submitCommands([
            .write(buffer: positions, offset: 0, words: xy.map(\.bitPattern)),
            .write(buffer: colors, offset: 0, words: rgb.map(\.bitPattern)),
            .write(buffer: indices, offset: 0, words: words),
            .renderIndexed(pipeline: pipeline, target: target, clear: clear, indexBuffer: binding,
                           draw: .init(indexCount: 3, firstIndex: first, baseVertex: -1), vertexBuffers: vertices),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .renderIndexed(pipeline: pipeline, target: target, clear: clear, indexBuffer: binding,
                           draw: .init(indexCount: 6, firstIndex: first, baseVertex: -1), vertexBuffers: vertices),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        try await device.destroyBuffer(indices)
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(snapshots[0].first == 0xff00ff00)
        #expect(snapshots[0].last == 0xffff0000)
        #expect(snapshots[1].first == 0xffff0000)
        #expect(snapshots[1].last == 0xffff0000)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
