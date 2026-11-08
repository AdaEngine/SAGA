import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let depthClipControlShader = """
    @vertex fn vs(@location(0) point: vec4f) -> @builtin(position) vec4f { return point; }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

@Test func depthClipControlRequiresAnAdvertisedFeatureBeforePipelineAllocation() async throws {
    let device = try await Device.make(backend: TestBackend())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(depthClipControlShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layout = VertexBufferLayout(stride: 16, attributes: [.init(location: 0, offset: 0, format: .float32x4)])
    #expect(!RenderPrimitiveState().unclippedDepth)
    await #expect(throws: GPUError.validation("Feature not advertised")) {
        try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                               vertexBuffers: [layout], primitive: .init(unclippedDepth: true))
    }
    await device.close()
}

#if canImport(Metal)
@Test func unclippedDepthKeepsSideClippingAndClampsInterpolatedDepthOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.depthClipControl])
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(depthClipControlShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let layout = VertexBufferLayout(stride: 16, attributes: [.init(location: 0, offset: 0, format: .float32x4)])
    let depthState = RenderDepthState(compare: .always)
    let clip = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                      vertexBuffers: [layout], depthState: depthState)
    let clamp = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], depthState: depthState, primitive: .init(unclippedDepth: true))
    let points: [Float] = [
        -1, -1, -1, 1,   3, -1, -1, 1,   -1, 3, -1, 1,
        -1, -1,  2, 1,   3, -1,  2, 1,   -1, 3,  2, 1,
         2, -1, -1, 1,   4, -1, -1, 1,    2, 3, -1, 1,
        -1, -1, -1, 1,   1, -1, 1.5, 1,   0, 1, 0.5, 1,
    ]
    let vertices = try await device.createBuffer(.init(size: UInt64(points.count * 4), usage: [.vertex, .copyDestination]))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
                                                      usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 256, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let raster = RenderRasterState(viewport: .init(x: 0, y: 0, width: 1, height: 1, minDepth: 0.25, maxDepth: 0.75))
    let draws: [(RenderPipeline, UInt32)] = [(clip, 0), (clamp, 0), (clamp, 3), (clip, 3), (clamp, 6), (clamp, 9)]
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: depth, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]
    var commands: [BufferCommand] = [.write(buffer: vertices, offset: 0, words: points.map(\.bitPattern))]
    for (pipeline, firstVertex) in draws {
        commands.append(.render(pipeline: pipeline, target: color, clear: clear, vertexCount: 3,
                                vertexBuffers: [.init(buffer: vertices)], firstVertex: firstVertex,
                                depth: .init(texture: depth, loadAction: .clear(0.5)), raster: raster))
        commands += capture
    }
    let result = try await device.submitCommands(commands).wait().flatMap { $0 }
    let blue: UInt32 = 0xffff0000
    let green: UInt32 = 0xff00ff00
    #expect(Array(result.dropLast()) == [
        blue, Float(0.5).bitPattern,
        green, Float(0.25).bitPattern,
        green, Float(0.75).bitPattern,
        blue, Float(0.5).bitPattern,
        blue, Float(0.5).bitPattern,
        green,
    ])
    let interpolatedDepth = Float(bitPattern: try #require(result.last))
    #expect(abs(interpolatedDepth - 0.4375) <= 1.0 / 1024)
    await device.close()
}
#endif
