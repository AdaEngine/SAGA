import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let renderStorageBufferShader = """
    struct Cell { @size(8) count: atomic<u32>, marker: u32, }
    struct Feedback { header: u32, cells: array<Cell>, }
    @group(0) @binding(0) var<storage, read> positions: array<vec2f>;
    @group(0) @binding(1) var<storage, read_write> feedback: Feedback;

    fn position(index: u32) -> vec2f {
        if arrayLength(&positions) != 3u { return vec2f(4.0f); }
        return positions[index];
    }

    @vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        return vec4f(position(index), 0.0f, 1.0f);
    }

    fn storeFeedback(index: u32) {
        atomicStore(&feedback.cells[index].count, arrayLength(&feedback.cells) * 100u + index);
    }

    @fragment fn fragmentMain(@builtin(position) point: vec4f) -> @location(0) vec4f {
        let pixel = vec2u(point.xy);
        storeFeedback(pixel.y * 2u + pixel.x);
        return vec4f(0.0f, 1.0f, 0.0f, 1.0f);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesRenderStorageBufferStages(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(renderStorageBufferShader)
    for entry in ["vertexMain", "fragmentMain"] {
        let shader = try compiler.compileRenderStage(source, entryPoint: entry, to: format)
        let binding = try #require(shader.storageBindings.first)
        #expect(shader.storageBindings.count == 1)
        #expect(binding.isReadOnly == (entry == "vertexMain"))
        #expect(binding.runtimeArrayStride == (entry == "vertexMain" ? 8 : 12))
        #expect(binding.runtimeArrayOffset == (entry == "vertexMain" ? 0 : 4))
        guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
        if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
    }
}

@Test func rejectsWritableRenderStorageInVertexStage() {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> value: u32;
        @vertex fn run() -> @builtin(position) vec4f {
            return vec4f(f32(value), 0.0f, 0.0f, 1.0f);
        }
        """)
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(source) }
}

#if canImport(Metal)
@Test func renderStorageBuffersUseRangedBindingsAndWriteFeedbackOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(renderStorageBufferShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vertexMain", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fragmentMain", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let positions = try await device.createBuffer(.init(size: 280, usage: [.storage, .copyDestination]))
        let feedback = try await device.createBuffer(.init(size: 308, usage: [.storage, .copyDestination, .copySource]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                           usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let bindings: [ComputeBindingSlot: ComputeBufferBinding] = [
            .init(binding: 0): .init(buffer: positions, offset: 256, size: 24),
            .init(binding: 1): .init(buffer: feedback, offset: 256, size: 52),
        ]
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3)])
        }
        var aliased = bindings
        aliased[.init(binding: 0)] = .init(buffer: feedback, offset: 256, size: 24)
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                                                     storageBuffers: aliased)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: positions, offset: 256, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            .write(buffer: feedback, offset: 256, words: [7, 0, 801, 10, 0, 802, 20, 0, 803, 30, 0, 804, 40]),
            .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
            .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, storageBuffers: bindings),
            .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                           indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                           storageBuffers: bindings),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 512),
            .copy(source: feedback, destination: read, byteCount: 308),
            .read(buffer: read, byteCount: 308),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.map(\.count) == [128, 77])
        let pixels = snapshots[0]
        #expect([pixels[0], pixels[1], pixels[64], pixels[65]] == [0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00])
        #expect(Array(snapshots[1][64..<77]) == [7, 400, 801, 10, 401, 802, 20, 402, 803, 30, 403, 804, 40])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
