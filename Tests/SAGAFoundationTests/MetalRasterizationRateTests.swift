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
      .gpuExclusive, arguments: [0 as UInt32, 1])
func rendersOnlyTheRequestedRateMapLayer(layer: UInt32) async throws {
    let backend = try MetalBackend.makeDevice()
    let metal = backend.metalDevice
    try #require(metal.supportsRasterizationRateMap(layerCount: 2))
    let descriptor = MTLRasterizationRateMapDescriptor(screenSize: MTLSize(width: 64, height: 64, depth: 1))
    descriptor.setLayer(MTLRasterizationRateLayerDescriptor(horizontal: [1], vertical: [1]), at: 0)
    descriptor.setLayer(MTLRasterizationRateLayerDescriptor(horizontal: [0.5], vertical: [1]), at: 1)
    let nativeMap = try #require(metal.makeRasterizationRateMap(descriptor: descriptor))
    let map = try MetalRasterizationRateBinding(map: nativeMap, layer: layer)
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
        let vertex = try compiler.compileMetalRasterizedVertexStage(shader, entryPoint: "vs", layer: layer)
        let fragment = try compiler.compileRenderStage(shader, entryPoint: "fs", to: .msl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let output = try await device.createTexture(.init(width: 64, height: 64, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource], arrayLayerCount: 2))
        let readback = try await device.createBuffer(.init(size: 32_768, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: output, colorLoad: .clear(.transparentBlack),
                    vertexCount: 3, raster: .init(rateMap: map)),
            .copyTextureToBuffer(source: output, destination: readback, bytesPerRow: 256, depth: 2),
            .read(buffer: readback, byteCount: 32_768),
        ])
        let images = try await ticket.wait()
        let words = try #require(images.first)
        let selected = Int(layer) * 4096
        let other = Int(1 - layer) * 4096
        let red: UInt32 = 0xff0000ff
        #expect(words[selected + 520] == red)
        #expect(words[other + 520] == 0)
        let expectedAtRightEdge = layer == 0 ? red : 0
        #expect(words[selected + 560] == expectedAtRightEdge)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
