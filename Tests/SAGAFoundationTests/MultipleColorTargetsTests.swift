import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let multipleTargetsShader = """
    struct FragmentOutput {
        @location(2) tag: vec4u,
        @location(0) color: vec4f,
    }
    struct FloatOutput {
        @location(0) first: vec4f,
        @location(2) second: vec4f,
    }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> FragmentOutput { return FragmentOutput(vec4u(17, 29, 33, 49), vec4f(1, 0, 0, 1)); }
    @fragment fn bothFloat() -> FloatOutput { return FloatOutput(vec4f(0), vec4f(1)); }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsMultipleFragmentColorLocations(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fragment = try compiler.compileRenderStage(.wgsl(multipleTargetsShader), entryPoint: "fs", to: format)
    #expect(fragment.colorOutputKinds == [0: .float, 2: .uint])
    guard case .text(_, let body) = fragment.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("@location(0)") && body.contains("@location(2)"))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl: #expect(body.contains("[[color(0)]]") && body.contains("[[color(2)]]"))
    case .hlsl: #expect(body.contains("SV_Target0") && body.contains("SV_Target2"))
    case .glsl:
        #expect(body.contains("layout(location = 0) out vec4 saga_color"))
        #expect(body.contains("layout(location = 2) out uvec4 saga_color_2"))
    default: Issue.record("Unexpected shader format")
    }
}

@Test func validatesColorTargetsAndRejectsAliasedAttachments() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(multipleTargetsShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    for targets: [UInt32: RenderColorTarget] in [[:], [2: .init(format: .rgba8Unorm)]] {
        await #expect(throws: GPUError.self) {
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                      additionalColorTargets: targets)
        }
    }
    let floatFragment = try compiler.compileRenderStage(source, entryPoint: "bothFloat", to: .msl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: floatFragment, colorFormat: .rgba8Unorm,
                                                         additionalColorTargets: [2: .init(format: .rgba8Unorm)])
    await #expect(throws: GPUError.validation("Color attachments exceed the portable bytes-per-sample limit")) {
        let formats = Dictionary(uniqueKeysWithValues: (UInt32(1)...4).map {
            ($0, RenderColorTarget(format: .rgba8Unorm))
        })
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: floatFragment, colorFormat: .rgba8Unorm,
                                                  additionalColorTargets: formats)
    }
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let wrongSize = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    for attachments: [UInt32: RenderColorAttachment] in [
        [:], [2: .init(texture: target)], [2: .init(texture: wrongSize)],
    ] {
        await #expect(throws: GPUError.self) {
            _ = try await device.submitCommands([
                .render(pipeline: pipeline, target: target, clear: .transparentBlack,
                        vertexCount: 3, additionalColors: attachments),
            ])
        }
    }
    await device.close()
}

#if canImport(Metal)
@Test(arguments: [false, true])
func rendersSparseMixedColorAttachmentsOnMetal(indexed: Bool) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(multipleTargetsShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            additionalColorTargets: [2: .init(format: .rgba8Uint, colorState: .init(writeMask: [.red, .blue]))])
        let primary = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let tag = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
            usage: [.renderAttachment, .copySource], mipLevelCount: 2))
        let view = try await device.createTextureView(texture: tag, .init(baseMipLevel: 1))
        let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let colors: [UInt32: RenderColorAttachment] = [
            2: .init(texture: view, loadAction: .clear(.init(red: 0, green: 5, blue: 0, alpha: 7))),
        ]
        let loaded: [UInt32: RenderColorAttachment] = [2: .init(texture: view)]
        let draw: BufferCommand = indexed
            ? .renderIndexed(pipeline: pipeline, target: primary, clear: .transparentBlack,
                             indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3),
                             additionalColors: loaded)
            : .render(pipeline: pipeline, target: primary, clear: .transparentBlack,
                      vertexCount: 3, additionalColors: loaded)
        let floatFragment = try compiler.compileRenderStage(source, entryPoint: "bothFloat", to: .msl)
        let constant = RenderBlendComponent(sourceFactor: .constant)
        let multisampled = try await device.createRenderPipeline(
            vertex: vertex, fragment: floatFragment, colorFormat: .rgba8Unorm, sampleCount: 4,
            additionalColorTargets: [2: .init(format: .rgba8Unorm,
                colorState: .init(blend: .init(color: constant, alpha: constant)))])
        let multisample = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment, sampleCount: 4)
        let msPrimary = try await device.createTexture(multisample)
        let msSecondary = try await device.createTexture(multisample)
        let ticket = try await device.submitCommands([
            .write(buffer: indices, offset: 0, words: [0, 1, 2]),
            .render(pipeline: pipeline, target: primary, clear: .transparentBlack,
                    vertexCount: 0, additionalColors: colors),
            draw,
            .copyTextureToBuffer(source: primary, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .copyTextureToBuffer(source: tag, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 4),
            .render(pipeline: multisampled, target: msPrimary, clear: .transparentBlack, vertexCount: 3,
                    blendConstant: .init(red: 0.25, green: 0.5, blue: 1, alpha: 1),
                    additionalColors: [2: .init(texture: msSecondary, resolve: .init(texture: primary))]),
            .copyTextureToBuffer(source: primary, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        ])
        #expect(try await ticket.wait() == [[0xff0000ff], [0x07210511], [0xffff8040]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
