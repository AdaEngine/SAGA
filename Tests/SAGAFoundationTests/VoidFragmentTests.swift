import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let voidFragmentShader = """
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point, 0.5, 1.0);
    }

    @fragment fn depth_only(@invariant @builtin(position) point: vec4f) {
        if point.x < 1.0 { discard; }
        if point.y < 1.0 { return; }
    }

    @fragment fn shade() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesFragmentWithNoReturnValue(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(voidFragmentShader), entryPoint: "depth_only", to: format)
    #expect(shader.colorOutputKinds.isEmpty)
    #expect(!shader.writesDepth)
    #expect(!shader.writesSampleMask)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader text"); return }
    #expect(source.contains("return;"))
    switch format {
    case .wgsl:
        #expect(!source.contains("->"))
        #expect(source.contains("discard;"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains("fragment void saga_fragment"))
        #expect(source.contains("discard_fragment();"))
    case .hlsl:
        #expect(source.contains("void saga_fragment"))
        #expect(!source.contains("SV_Target"))
    case .glsl:
        #expect(source.contains("void main()"))
        #expect(!source.contains("layout(location = 0) out"))
    default: Issue.record("Unexpected target")
    }
}

#if canImport(Metal)
@Test func voidFragmentWritesRasterDepthWithDiscardAndPipelineSampleMaskOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(voidFragmentShader)
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let depthFragment = try compiler.compileRenderStage(source, entryPoint: "depth_only", to: .msl)
    let shade = try compiler.compileRenderStage(source, entryPoint: "shade", to: .msl)
    let write = try await device.createRenderPipeline(vertex: vertex, fragment: depthFragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], depthState: .init(compare: .always))
    let masked = try await device.createRenderPipeline(vertex: vertex, fragment: depthFragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], depthState: .init(compare: .always), sampleMask: 0)
    let testDepth = try await device.createRenderPipeline(vertex: vertex, fragment: shade, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], depthState: .init(compare: .equal, writeEnabled: false))
    #expect(!write.writesColor)
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 2, height: 2, format: .depth32Float, usage: .renderAttachment))
    let readback = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
    ]
    var commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
    ]
    for pipeline in [write, masked] {
        commands.append(.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                                vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth, loadAction: .clear(1))))
        commands += capture
        commands.append(.render(pipeline: testDepth, target: target, colorLoad: .load, vertexCount: 3,
                                vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth, loadAction: .load)))
        commands += capture
    }
    let snapshots = try await device.submitCommands(commands).wait()
    #expect(snapshots.map { [$0[0], $0[1], $0[64], $0[65]] } == [
        [0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000],
        [0xffff0000, 0xff00ff00, 0xffff0000, 0xff00ff00],
        [0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000],
        [0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000],
    ])
    await device.close()
}
#endif
