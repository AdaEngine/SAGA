import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let renderTextureShader = """
    @group(2) @binding(5) var image: texture_2d<f32>;
    @group(1) @binding(4) var auxiliary: texture_2d<f32>;
    @group(3) @binding(6) var unused: texture_2d<f32>;
    @group(2) @binding(8) var<uniform> gain: f32;

    fn readImage(x: i32) -> vec4f { return textureLoad(image, vec2i(x, 0i), 0i); }

    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point.x + readImage(0i).r, point.y, 0.0f, 1.0f);
    }

    @fragment fn fs() -> @location(0) vec4f {
        let blue = textureLoad(auxiliary, vec2u(0u), 0u).b;
        return vec4f(readImage(1i).g * gain, blue, 0.0f, 1.0f);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsSampledTexturesReachableFromEachRenderStage(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shared = IRResourceBinding(group: 2, binding: 5)
    let auxiliary = IRResourceBinding(group: 1, binding: 4)
    for entry in ["vs", "fs"] {
        let stage = try compiler.compileRenderStage(.wgsl(renderTextureShader), entryPoint: entry, to: format)
        let vertex = entry == "vs"
        guard case .text(_, let source) = stage.artifact else { Issue.record("Expected source"); return }
        switch format {
        case .msl:
            #expect(source.contains("saga_texture_0 [[texture(\(vertex ? 0 : 1))]]"))
            #expect(source.contains(".read("))
        case .hlsl:
            #expect(source.contains("register(t5, space2)"))
            #expect(source.contains("register(t4, space1)") == !vertex)
            #expect(source.contains(".Load("))
        case .glsl:
            #expect(source.contains("layout(binding=1) uniform sampler2D saga_texture_0"))
            #expect(source.contains("layout(binding=0) uniform sampler2D saga_texture_1") == !vertex)
            #expect(source.contains("texelFetch("))
        case .wgsl:
            #expect(source.contains("@group(2) @binding(5)"))
            #expect(source.contains("@group(1) @binding(4)") == !vertex)
            #expect(source.contains("textureLoad("))
            _ = try compiler.parseAndValidate(.wgsl(source))
        default: Issue.record("Unexpected target")
        }
        #expect(stage.sampledTextures.map(\.binding) == (vertex ? [shared] : [shared, auxiliary]))
    }
}

#if canImport(Metal)
@Test func drawsWithSharedSampledTexturesAndRejectsAliasedAttachmentsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(.wgsl(renderTextureShader), entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(.wgsl(renderTextureShader), entryPoint: "fs", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
        let conflictingFragment = try compiler.compileRenderStage(.wgsl("""
            @group(2) @binding(5) var<uniform> color: vec4f;
            @fragment fn fs() -> @location(0) vec4f { return color; }
            """), entryPoint: "fs", to: .msl)
        await #expect(throws: GPUError.validation("Render stages disagree on the resource type of a binding")) {
            try await device.createRenderPipeline(vertex: vertex, fragment: conflictingFragment,
                                                  colorFormat: .rgba8Unorm, vertexBuffers: layouts)
        }
        let upload = try await device.createBuffer(.init(size: 12, usage: [.copySource, .copyDestination]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let gain = try await device.createBuffer(.init(size: 4, usage: [.uniform, .copyDestination]))
        let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                        usage: [.copyDestination, .textureBinding]))
        let aux = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8Unorm,
                                                      usage: [.copyDestination, .textureBinding]))
        let fresh = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8Unorm, usage: .textureBinding))
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource, .textureBinding]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let shared = ComputeBindingSlot(group: 2, binding: 5)
        let auxiliary = ComputeBindingSlot(group: 1, binding: 4)
        let uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [.init(group: 2, binding: 8): .init(buffer: gain)]
        let streams = [VertexBufferBinding(buffer: vertices)]
        func draw(_ textures: [ComputeBindingSlot: ResourceID]) -> BufferCommand {
            .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                    vertexBuffers: streams, uniforms: uniforms, textures: textures)
        }
        await #expect(throws: GPUError.validation("Render attachment cannot be bound as a sampled texture")) {
            try await device.submitCommands([draw([shared: target, auxiliary: aux])])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([draw([shared: image])])
        }
        let drawWithTextures = draw([shared: image, auxiliary: aux])
        #expect(drawWithTextures.resources.contains(image))
        #expect(drawWithTextures.resources.contains(aux))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ff00, 0xff0000ff]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(width: 2, height: 1)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: aux, region: .init(width: 1, height: 1)),
            .write(buffer: gain, offset: 0, words: [Float(1).bitPattern]),
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
            drawWithTextures,
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                           indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                           vertexBuffers: streams, uniforms: uniforms, textures: [shared: image, auxiliary: fresh]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        try await device.destroyTexture(image)
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2 && snapshots.allSatisfy { $0.count == 456 })
        let pixels = snapshots.flatMap { [$0[4 * 64 + 1], $0[4 * 64 + 6]] }
        #expect(pixels == [0xffff0000, 0xff00ffff, 0xffff0000, 0xff0000ff])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
