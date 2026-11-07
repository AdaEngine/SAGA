import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let fragmentDepthShader = """
    struct FragmentOut {
        @builtin(frag_depth) depth: f32,
        @builtin(sample_mask) mask: u32,
        @location(0) color: vec4f,
    }
    @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        if index == 0u { return vec4f(-1, -1, 0.75, 1); }
        if index == 1u { return vec4f(3, -1, 0.75, 1); }
        return vec4f(-1, 3, 0.75, 1);
    }
    @fragment fn colorAndDepth() -> FragmentOut {
        return FragmentOut(0.25, 1u, vec4f(1, 0, 0, 1));
    }
    @fragment fn depthOnly() -> @builtin(frag_depth) f32 { return 2.0; }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsFragmentDepthAcrossTargets(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(fragmentDepthShader)
    let combined = try compiler.compileRenderStage(source, entryPoint: "colorAndDepth", to: format)
    let depth = try compiler.compileRenderStage(source, entryPoint: "depthOnly", to: format)
    #expect(combined.colorOutputKind == .float)
    #expect(depth.colorOutputKind == nil)
    #expect(combined.writesDepth && depth.writesDepth)
    guard case .text(_, let both) = combined.artifact, case .text(_, let direct) = depth.artifact else {
        Issue.record("Expected source"); return
    }
    switch format {
    case .wgsl:
        #expect(both.contains("@builtin(frag_depth)") && both.contains("@builtin(sample_mask)"))
        #expect(direct.contains("-> @builtin(frag_depth) f32"))
        _ = try compiler.parseAndValidate(.wgsl(both))
        _ = try compiler.parseAndValidate(.wgsl(direct))
    case .msl:
        #expect(both.contains("[[depth(any)]]") && both.contains("[[sample_mask]]"))
        #expect(direct.contains("[[depth(any)]]"))
    case .hlsl:
        #expect(both.contains("SV_Depth") && both.contains("SV_Coverage"))
        #expect(direct.contains(": SV_Depth"))
    case .glsl:
        #expect(both.contains("gl_FragDepth =") && both.contains("gl_SampleMask[0] ="))
        #expect(direct.contains("gl_FragDepth ="))
    default: Issue.record("Unexpected shader target")
    }
}

#if canImport(Metal)
@Test func fragmentDepthControlsTestingAndStoresClampedDepthOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(fragmentDepthShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let combined = try compiler.compileRenderStage(source, entryPoint: "colorAndDepth", to: .msl)
        let onlyDepth = try compiler.compileRenderStage(source, entryPoint: "depthOnly", to: .msl)
        await #expect(throws: GPUError.validation("Fragment depth output requires a depth attachment")) {
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: combined, colorFormat: .rgba8Unorm)
        }
        let writesColor = try await device.createRenderPipeline(
            vertex: vertex, fragment: combined, colorFormat: .rgba8Unorm, depthState: .init(compare: .less))
        let writesDepth = try await device.createRenderPipeline(
            vertex: vertex, fragment: onlyDepth, colorFormat: .rgba8Unorm, depthState: .init(compare: .always))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let ticket = try await device.submitCommands([
            .render(pipeline: writesColor, target: color, clear: blue, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .clear(0.5))),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .render(pipeline: writesDepth, target: color, clear: blue, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .load)),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        ])
        #expect(try await ticket.wait() == [[0xff0000ff], [Float(0.25).bitPattern], [0xffff0000], [Float(1).bitPattern]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
