import Testing
import SAGACompiler
import SAGATypes
import SAGACore
import SAGAMetal

private let sparseLayouts: [VertexBufferLayout] = [
    .init(stride: 0, attributes: []),
    .init(stride: 8, attributes: [.init(location: 4, offset: 0, format: .float32x2)]),
    .init(stride: 0, attributes: []),
    .init(stride: 0, attributes: [.init(location: 1, offset: 0, format: .float32x3)]),
]

@Test func sparseVertexSlotsValidateOnlyActiveBindings() async throws {
    let backend = RenderTestBackend()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "v", to: .msl)
        let fragment = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "f", to: .msl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: sparseLayouts)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .renderAttachment))
        let positions = try await device.createBuffer(.init(size: 24, usage: .vertex))
        let color = try await device.createBuffer(.init(size: 12, usage: .vertex))
        let bindings: [VertexBufferBinding] = [.init(buffer: positions, slot: 1), .init(buffer: color, slot: 3), .init(buffer: color, slot: 7)]
        let ticket = try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3, vertexBuffers: bindings)])
        await backend.release.release()
        _ = try await ticket.wait()
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 4, vertexBuffers: bindings)])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3, vertexBuffers: [.init(buffer: positions, slot: 1)])])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3, vertexBuffers: [.init(buffer: positions, slot: 1), .init(buffer: color, slot: 1)])])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3, vertexBuffers: [.init(buffer: positions, slot: .max), .init(buffer: color, slot: 3)])])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3, vertexBuffers: [.init(buffer: positions, slot: 1), .init(buffer: color, slot: 3), .init(buffer: color, slot: 8)])])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3, vertexBuffers: [.init(buffer: positions, slot: 1), .init(buffer: color, slot: 3), .init(buffer: color, offset: 2, slot: 7)])])
        }
    } catch {
        await backend.release.release()
        await device.close()
        throw error
    }
    await device.close()
}

#if canImport(Metal)
@Test(arguments: [false, true]) func metalSparseVertexSlotsRender(pulling: Bool) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let vertex = try pulling
        ? compiler.compileMetalVertexStage(vertexBufferShader, entryPoint: "v", vertexBuffers: sparseLayouts)
        : compiler.compileRenderStage(vertexBufferShader, entryPoint: "v", to: .msl)
    let fragment = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "f", to: .msl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: sparseLayouts)
    let positions = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let color = try await device.createBuffer(.init(size: 12, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
    let xy: [Float] = [-1, 1, 1, 1, -1, -1]
    let rgb: [Float] = [0, 1, 0]
    let ticket = try await device.submitCommands([
        .write(buffer: positions, offset: 0, words: xy.map(\.bitPattern)),
        .write(buffer: color, offset: 0, words: rgb.map(\.bitPattern)),
        .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3, vertexBuffers: [.init(buffer: positions, slot: 1), .init(buffer: color, slot: 3), .init(buffer: color, slot: 7)]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
    ])
    let pixels = try #require(try await ticket.wait().first)
    #expect(pixels.first == 0xff00ff00)
    #expect(pixels.last == 0xff000000)
    await device.close()
}
#endif
