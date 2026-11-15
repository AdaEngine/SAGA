#if canImport(Metal)
import Metal
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

@MainActor
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRasterizationRateMap(layerCount: 2) == true),
      .gpuExclusive, arguments: [(0, 1), (0, 2), (1, 2)])
func rendersOnlyTheRequestedRateMapLayer(layer: Int, layers: Int) async throws {
    let backend = try MetalBackend.makeDevice()
    let metal = backend.metalDevice
    try #require(metal.supportsRasterizationRateMap(layerCount: 2))
    let descriptor = MTLRasterizationRateMapDescriptor(screenSize: MTLSize(width: 64, height: 64, depth: 1))
    for index in 0..<layers {
        let rate: Float = layers == 1 || index == 1 ? 0.5 : 1
        descriptor.setLayer(MTLRasterizationRateLayerDescriptor(horizontal: [rate], vertical: [1]), at: index)
    }
    let nativeMap = try #require(metal.makeRasterizationRateMap(descriptor: descriptor))
    let map = try MetalRasterizationRateBinding(map: nativeMap, layer: UInt32(layer))
    let shader = ShaderInput.wgsl("""
        struct Output { @builtin(position) position: vec4f, @location(0) color: vec4f }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> Output {
            let pos = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return Output(vec4f(pos[i], 0, 1), vec4f(1, 0, 0, 1));
        }
        @fragment fn fs(@location(0) color: vec4f) -> @location(0) vec4f { return color; }
        """)
    let device = try await Device.make(backend: backend, requiredFeatures: [.rasterizationRateMap])
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = if let vertexLayer = map.vertexLayer {
            try compiler.compileMetalRasterizedVertexStage(shader, entryPoint: "vs", layer: vertexLayer)
        } else {
            try compiler.compileRenderStage(shader, entryPoint: "vs", to: .msl)
        }
        let fragment = try compiler.compileRenderStage(shader, entryPoint: "fs", to: .msl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let output = try await device.createTexture(.init(width: 64, height: 64, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource], arrayLayerCount: UInt32(layers)))
        let byteCount = UInt64(16_384 * layers)
        let readback = try await device.createBuffer(.init(size: byteCount, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: output, colorLoad: .clear(.transparentBlack),
                    vertexCount: 3, raster: .init(rateMap: map)),
            .copyTextureToBuffer(source: output, destination: readback, bytesPerRow: 256, depth: UInt32(layers)),
            .read(buffer: readback, byteCount: byteCount),
        ])
        let images = try await ticket.wait()
        let words = try #require(images.first)
        let selected = layer * 4096
        let red: UInt32 = 0xff0000ff
        #expect(words[selected + 520] == red)
        if layers == 2 { #expect(words[(1 - layer) * 4096 + 520] == 0) }
        let expectedAtRightEdge = layers == 2 && layer == 0 ? red : 0
        #expect(words[selected + 560] == expectedAtRightEdge)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
