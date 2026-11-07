import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let fragmentRasterShader = """
    struct RasterInput {
        @builtin(position) pixel: vec4f,
    }
    @vertex fn vs(@location(0) clip: vec2f) -> @builtin(position) vec4f {
        return vec4f(clip, 0.5, 2.0);
    }
    @fragment fn fs(input: RasterInput, @builtin(front_facing) front: bool) -> @location(0) vec4u {
        let p = input.pixel;
        let x = select(0u, 1u, p.x == 0.5) + select(0u, 2u, p.x == 1.5);
        let y = select(0u, 1u, p.y == 0.5) + select(0u, 2u, p.y == 1.5);
        let zw = select(0u, 37u, p.z == 0.25) + select(0u, 64u, p.w == 0.5);
        return vec4u(x, y, zw, select(0u, 1u, front));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsFragmentPositionAndFrontFacingAcrossTargets(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fragment = try compiler.compileRenderStage(.wgsl(fragmentRasterShader), entryPoint: "fs", to: format)
    #expect(fragment.varyings.isEmpty)
    guard case .text(_, let code) = fragment.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(code.contains("@builtin(position)") && code.contains("@builtin(front_facing)"))
        _ = try compiler.parseAndValidate(.wgsl(code))
    case .msl:
        #expect(code.contains("[[position]]") && code.contains("[[front_facing]]"))
    case .hlsl:
        #expect(code.contains("SV_Position") && code.contains("SV_IsFrontFace"))
        #expect(code.contains("1.0f / saga_input.saga_position.w"))
    case .glsl:
        #expect(code.contains("gl_FragCoord") && code.contains("gl_FrontFacing"))
    default: Issue.record("Unexpected shader output")
    }
}

#if canImport(Metal)
@Test func readsFragmentCoordinatesAndUsesPipelineWindingAndCullingOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(fragmentRasterShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let defaults = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, vertexBuffers: layouts)
        let clockwise = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, vertexBuffers: layouts,
            primitive: .init(frontFace: .clockwise, cullMode: .back))
        let cullFront = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, vertexBuffers: layouts,
            primitive: .init(cullMode: .front))
        let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
            usage: [.renderAttachment, .copySource]))
        let vertices = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        let triangles: [Float] = [-2, -2, 6, -2, -2, 6, -2, -2, -2, 6, 6, -2]
        var commands: [BufferCommand] = [.write(buffer: vertices, offset: 0, words: triangles.map(\.bitPattern))]
        for (pipeline, firstVertex) in [(defaults, 0), (defaults, 3), (clockwise, 0), (clockwise, 3), (cullFront, 0)] {
            commands += [
                .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3,
                        vertexBuffers: [.init(buffer: vertices)], firstVertex: UInt32(firstVertex)),
                .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
                .read(buffer: read, byteCount: 264),
            ]
        }
        let ticket = try await device.submitCommands(commands)
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 5 && snapshots.allSatisfy { $0.count == 66 })
        let pixels = snapshots.map { [$0[0], $0[1], $0[64], $0[65]] }
        let front: [UInt32] = [0x01650101, 0x01650102, 0x01650201, 0x01650202]
        let back = front.map { $0 & 0x00ff_ffff }
        let culled: [UInt32] = [0, 0, 0, 0]
        #expect(pixels == [front, back, culled, front, culled])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
