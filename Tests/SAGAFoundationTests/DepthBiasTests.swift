import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let depthBiasShader = """
    @vertex fn flatDepth(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0.5, 1); }
        if i == 1u { return vec4f(3, -1, 0.5, 1); }
        return vec4f(-1, 3, 0.5, 1);
    }
    @vertex fn slope(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0.25, 1); }
        if i == 1u { return vec4f(3, -1, 0.75, 1); }
        return vec4f(-1, 3, 0.25, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
    """

@Test func rejectsDepthBiasForLinesAndNonfiniteFactors() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(depthBiasShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "flatDepth", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    await #expect(throws: GPUError.validation("Depth bias requires a triangle topology")) {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            depthState: .init(bias: .init(clamp: 0.125)), primitive: .init(topology: .lineList))
    }
    await #expect(throws: GPUError.validation("Depth bias slope scale and clamp must be finite")) {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            depthState: .init(bias: .init(slopeScale: .infinity)))
    }
    await device.close()
}

#if canImport(Metal)
@Test func appliesConstantSlopeAndClampedDepthBiasOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(depthBiasShader)
        let flat = try compiler.compileRenderStage(source, entryPoint: "flatDepth", to: .msl)
        let slope = try compiler.compileRenderStage(source, entryPoint: "slope", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let cases: [(RenderStageArtifact, RenderDepthBiasState, Double)] = [
            (flat, .init(), 0.53125),
            (flat, .init(constant: 1 << 20), 0.53125),
            (flat, .init(constant: 1 << 20), 1),
            (flat, .init(constant: -(1 << 20)), 1),
            (slope, .init(slopeScale: 1, clamp: 0.03125), 1),
            (slope, .init(slopeScale: -1, clamp: -0.03125), 1),
        ]
        var commands: [BufferCommand] = []
        for (vertex, bias, clear) in cases {
            let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                                 depthState: .init(bias: bias))
            commands += [
                .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                        depth: .init(texture: depth, loadAction: .clear(clear))),
                .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
                .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            ]
        }
        let ticket = try await device.submitCommands(commands)
        let red: UInt32 = 0xff0000ff
        let expected: [[UInt32]] = [
            [red], [Float(0.5).bitPattern],
            [0], [Float(0.53125).bitPattern],
            [red], [Float(0.5625).bitPattern],
            [red], [Float(0.4375).bitPattern],
            [red], [Float(0.40625).bitPattern],
            [red], [Float(0.34375).bitPattern],
        ]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
