import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let renderStorageShader = """
    @group(2) @binding(4) var destination: texture_storage_2d<rgba8unorm, write>;
    struct VertexOut {
        @builtin(position) position: vec4f,
        @location(0) x: f32,
    }
    @vertex fn vs(@location(0) point: vec2f) -> VertexOut {
        return VertexOut(vec4f(point, 0, 1), point.x);
    }
    fn writeColor(x: i32, color: vec4f) {
        textureStore(destination, vec2i(x, 0), color);
    }
    @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
        if x < 0.0 { writeColor(1, vec4f(1, 0, 0, 1)); }
        else { writeColor(i32(textureDimensions(destination).x) - 2i, vec4f(0, 0, 1, 1)); }
        return vec4f(0, 1, 0, 1);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsFragmentStorageTextureWritesThroughHelpers(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(renderStorageShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
    #expect(vertex.storageTextures.isEmpty)
    let binding = try #require(fragment.storageTextures.first)
    #expect(fragment.storageTextures.count == 1)
    #expect(binding.binding == .init(group: 2, binding: 4) && binding.format == .rgba8Unorm)
    guard case .text(_, let body) = fragment.artifact else { Issue.record("Expected fragment source"); return }
    switch format {
    case .msl:
        #expect(body.contains("access::write> saga_texture_0 [[texture(24)]]"))
        #expect(body.contains(".write("))
    case .hlsl:
        #expect(body.contains("RWTexture2D<unorm float4>") && body.contains("register(u4, space2)"))
    case .glsl:
        #expect(body.contains("layout(rgba8, binding=0) writeonly uniform image2D") && body.contains("imageStore("))
    case .wgsl:
        #expect(body.contains("textureStore(") && body.contains("textureDimensions("))
        _ = try compiler.parseAndValidate(.wgsl(body))
    default: Issue.record("Unexpected format")
    }
}

@Test func rejectsStorageTextureWritesFromVertexStage() throws {
    let source = """
        @group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
        fn write() { textureStore(image, vec2u(0), vec4f(1)); }
        @vertex fn main() -> @builtin(position) vec4f { write(); return vec4f(1); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Writable storage texture accepted in vertex stage")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .invalidControlFlow)
    }
}

#if canImport(Metal)
@Test func fragmentStoresInSiblingMipWhileDrawingColorOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(renderStorageShader)
        let pipeline = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm, vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
        let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource, .storageBinding], mipLevelCount: 2))
        let storage = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
        let attachment = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
        let overlapping = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
        let storageIndexed = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: [.copySource, .storageBinding]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        let slot = ComputeBindingSlot(group: 2, binding: 4)
        let streams = [VertexBufferBinding(buffer: vertices)]
        func draw(_ output: ResourceID) -> BufferCommand {
            .render(pipeline: pipeline, target: attachment, clear: clear, vertexCount: 3,
                    vertexBuffers: streams, textures: [slot: output])
        }
        await #expect(throws: GPUError.validation("Aliased writable render textures")) {
            try await device.submitCommands([draw(overlapping)])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([draw(image)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
            draw(storage),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 272),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 8),
            .renderIndexed(pipeline: pipeline, target: attachment, clear: clear,
                           indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                           vertexBuffers: streams, textures: [slot: storageIndexed]),
            .copyTextureToBuffer(source: storageIndexed, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 272),
        ])
        let expectedStorage: [UInt32] = [0, 0xff0000ff, 0xffff0000, 0] + Array(repeating: 0, count: 64)
        let expected: [[UInt32]] = [expectedStorage, [0xff00ff00, 0xff00ff00], expectedStorage]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
