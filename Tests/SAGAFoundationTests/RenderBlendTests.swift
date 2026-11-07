import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let blendShader = """
    @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        if index == 0u { return vec4f(-1, -1, 0, 1); }
        if index == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn base() -> @location(0) vec4f { return vec4f(0, 0.5, 1, 1); }
    @fragment fn overlay() -> @location(0) vec4f { return vec4f(1, 0, 0, 0.5); }
    @fragment fn integerColor() -> @location(0) vec4u { return vec4u(1); }
    """

@Test func validatesColorBlendStateBeforeAllocatingPipeline() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(blendShader), entryPoint: "vs", to: .msl)
    let float = try compiler.compileRenderStage(.wgsl(blendShader), entryPoint: "base", to: .msl)
    let integer = try compiler.compileRenderStage(.wgsl(blendShader), entryPoint: "integerColor", to: .msl)
    await #expect(throws: GPUError.validation("Blending requires a blendable color attachment with a fragment color output")) {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: integer, colorFormat: .rgba8Uint,
                                                  colorState: .init(blend: .init()))
    }
    await #expect(throws: GPUError.validation("Min and max blending require one for both factors")) {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: float, colorFormat: .rgba8Unorm,
            colorState: .init(blend: .init(alpha: .init(operation: .max, sourceFactor: .sourceAlpha))))
    }
    await #expect(throws: GPUError.validation("Invalid color write mask")) {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: float, colorFormat: .rgba8Unorm,
                                                  colorState: .init(writeMask: .init(rawValue: 0x10)))
    }
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: float, colorFormat: .rgba8Unorm, colorState: .init(blend: .init()))
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    await #expect(throws: GPUError.validation("Blend constant must contain finite float values")) {
        _ = try await device.submitCommands([
            .render(pipeline: pipeline, target: target, colorLoad: .load, vertexCount: 3,
                    blendConstant: .init(red: .nan, green: 0, blue: 0, alpha: 0)),
        ])
    }
    await device.close()
}

#if canImport(Metal)
@Test func loadsColorBlendsAndMasksChannelsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(blendShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let base = try compiler.compileRenderStage(source, entryPoint: "base", to: .msl)
        let overlay = try compiler.compileRenderStage(source, entryPoint: "overlay", to: .msl)
        let paint = try await device.createRenderPipeline(vertex: vertex, fragment: base, colorFormat: .rgba8Unorm)
        let blend = try await device.createRenderPipeline(
            vertex: vertex, fragment: overlay, colorFormat: .rgba8Unorm,
            colorState: .init(blend: .init(color: .init(sourceFactor: .sourceAlpha, destinationFactor: .oneMinusSourceAlpha)),
                              writeMask: [.red, .green, .alpha]))
        let constantBlend = try await device.createRenderPipeline(
            vertex: vertex, fragment: overlay, colorFormat: .rgba8Unorm,
            colorState: .init(blend: .init(color: .init(operation: .reverseSubtract, sourceFactor: .constant, destinationFactor: .one),
                                          alpha: .init(operation: .min, sourceFactor: .one, destinationFactor: .one))))
        let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource], mipLevelCount: 2))
        let view = try await device.createTextureView(texture: texture, .init(baseMipLevel: 1))
        let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .render(pipeline: paint, target: view, colorLoad: .load, vertexCount: 0),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 4),
            .render(pipeline: paint, target: view, clear: .init(red: 0, green: 0, blue: 0, alpha: 0), vertexCount: 3),
            .write(buffer: indices, offset: 0, words: [0, 1, 2]),
            .renderIndexed(pipeline: blend, target: view, colorLoad: .load,
                           indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3)),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 4),
            .render(pipeline: constantBlend, target: view, clear: .init(red: 1, green: 0.5, blue: 0.25, alpha: 1),
                    vertexCount: 3, blendConstant: .init(red: 0.25, green: 0, blue: 0, alpha: 0)),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 4),
        ])
        #expect(try await ticket.wait() == [[0], [0x80ff4080], [0x804080bf]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
