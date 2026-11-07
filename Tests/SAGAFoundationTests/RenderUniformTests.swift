import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let renderUniformShader = """
    struct DrawConfig { offset: vec2f, color: vec4f }
    @group(3) @binding(11) var<uniform> cfg: DrawConfig;
    @group(0) @binding(0) var<uniform> jitter: vec2f;
    @group(1) @binding(1) var<uniform> unused: vec4f;
    fn tint() -> vec4f { return cfg.color; }
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point.x + cfg.offset.x + jitter.x, point.y + cfg.offset.y + jitter.y, 0.0f, 1.0f);
    }
    @fragment fn fs() -> @location(0) vec4f { return tint(); }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsReachableUniformsForBothGraphicsStages(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for entry in ["vs", "fs"] {
        let artifact = try compiler.compileRenderStage(.wgsl(renderUniformShader), entryPoint: entry, to: format)
        #expect(artifact.uniformBindings.map(\.minimumByteSize) == (entry == "vs" ? [32, 8] : [32]))
        let shared = IRResourceBinding(group: 3, binding: 11)
        #expect(artifact.uniformBindings.map(\.binding) == (entry == "vs" ? [shared, .init(group: 0, binding: 0)] : [shared]))
        guard case .text(_, let source) = artifact.artifact else { Issue.record("Expected source"); return }
        switch format {
        case .msl: #expect(source.contains("constant uint* saga_buffer_0 [[buffer(\(entry == "vs" ? 10 : 9))]]"))
        case .hlsl: #expect(source.contains("register(b11, space3)"))
        case .glsl: #expect(source.contains("std140") && source.contains("layout(offset = 16) vec4"))
        case .wgsl: _ = try compiler.parseAndValidate(.wgsl(source))
        default: Issue.record("Unexpected format")
        }
    }
}

#if canImport(Metal)
@Test func renderUniformSliceFeedsVertexAndFragmentStagesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(.wgsl(renderUniformShader), entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(.wgsl(renderUniformShader), entryPoint: "fs", to: .msl)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
        let uniform = try await device.createBuffer(.init(size: 288, usage: [.uniform, .copyDestination]))
        let jitter = try await device.createBuffer(.init(size: 8, usage: [.uniform, .copyDestination]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                           usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 1, green: 0, blue: 0, alpha: 1)
        let slot = ComputeBindingSlot(group: 3, binding: 11)
        let bindings: [ComputeBindingSlot: ComputeBufferBinding] = [
            slot: .init(buffer: uniform, offset: 256, size: 32),
            .init(binding: 0): .init(buffer: jitter),
        ]
        let streams = [VertexBufferBinding(buffer: vertices)]
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([
                .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3, vertexBuffers: streams)
            ])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([
                .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3, vertexBuffers: streams,
                        uniforms: [slot: .init(buffer: uniform, offset: 256, size: 16), .init(binding: 0): .init(buffer: jitter)])
            ])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: uniform, offset: 256, words: [Float(0.75).bitPattern, 0, 0, 0, 0, Float(1).bitPattern, 0, Float(1).bitPattern]),
            .write(buffer: jitter, offset: 0, words: [Float(0.25).bitPattern, 0]),
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
            .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3,
                    vertexBuffers: streams, uniforms: bindings),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                           indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                           vertexBuffers: streams, uniforms: bindings),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2 && snapshots.allSatisfy { $0.count == 456 })
        for words in snapshots {
            #expect(words[4 * 64 + 1] == 0xff0000ff)
            #expect(words[4 * 64 + 6] == 0xff00ff00)
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
