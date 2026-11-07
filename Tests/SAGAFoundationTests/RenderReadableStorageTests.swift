import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let renderReadableStorageShader = """
    @group(2) @binding(5) var mask: texture_storage_2d<rgba8unorm, read>;
    @group(1) @binding(4) var image: texture_storage_2d<r32uint, read_write>;

    struct VertexOut {
        @builtin(position) position: vec4f,
        @location(0) x: f32,
    }
    @vertex fn vs(@location(0) point: vec2f) -> VertexOut {
        let x = point.x + textureLoad(mask, vec2u(0u)).r;
        return VertexOut(vec4f(x, point.y, 0.0, 1.0), x);
    }
    @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
        let coords = vec2u(0u);
        let old = textureLoad(image, coords);
        if x > 0.0 { textureStore(image, coords, old + vec4u(5u)); }
        let red = select(0.0, 1.0, old.r == 10u);
        return vec4f(red, textureLoad(mask, coords).g, 0.0, 1.0);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsRenderStorageAccessAndSharedBindings(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(renderReadableStorageShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
    #expect(vertex.storageTextures.map(\.access) == [.load])
    #expect(fragment.storageTextures.map(\.access) == [.load, [.load, .store]])
    guard case .text(_, let vs) = vertex.artifact, case .text(_, let fs) = fragment.artifact else {
        Issue.record("Expected source")
        return
    }
    switch format {
    case .wgsl:
        _ = try compiler.parseAndValidate(.wgsl(vs))
        _ = try compiler.parseAndValidate(.wgsl(fs))
    case .msl:
        #expect(vs.contains("saga_texture_0 [[texture(16)]]"))
        #expect(fs.contains("saga_texture_0 [[texture(17)]]"))
        #expect(fs.contains("texture2d<uint, access::read_write>"))
    case .hlsl:
        #expect(vs.contains("register(u5, space2)") && fs.contains("register(u5, space2)"))
        #expect(fs.contains("RWTexture2D<uint>"))
    case .glsl:
        #expect(vs.contains("layout(rgba8, binding=1) readonly uniform image2D"))
        #expect(fs.contains("layout(rgba8, binding=1) readonly uniform image2D"))
        #expect(fs.contains("layout(r32ui, binding=0) uniform uimage2D"))
    default: Issue.record("Unexpected format")
    }
}

#if canImport(Metal)
@Test func readsSharedStorageInRenderStagesAndUpdatesIntegerTexelsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(renderReadableStorageShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
        let conflicting = try compiler.compileRenderStage(.wgsl("""
            @group(2) @binding(5) var mask: texture_storage_2d<r32uint, read>;
            struct VertexOut {
                @builtin(position) position: vec4f,
                @location(0) x: f32,
            }
            @vertex fn vs(@location(0) point: vec2f) -> VertexOut {
                let x = point.x + f32(textureLoad(mask, vec2u(0u)).r);
                return VertexOut(vec4f(x, point.y, 0.0, 1.0), x);
            }
            """), entryPoint: "vs", to: .msl)
        await #expect(throws: GPUError.validation("Render stages disagree on the format or access of a storage texture binding")) {
            try await device.createRenderPipeline(vertex: conflicting, fragment: fragment, colorFormat: .rgba8Unorm,
                                                  vertexBuffers: layouts)
        }
        let mask = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.copyDestination, .storageBinding]))
        let image = try await device.createTexture(.init(width: 1, height: 1, format: .r32Uint,
            usage: [.copySource, .copyDestination, .storageBinding]))
        let target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.copySource, .renderAttachment]))
        let upload = try await device.createBuffer(.init(size: 8, usage: [.copySource, .copyDestination]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let maskSlot = ComputeBindingSlot(group: 2, binding: 5)
        let imageSlot = ComputeBindingSlot(group: 1, binding: 4)
        func draw(_ resource: ResourceID) -> BufferCommand {
            .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)], textures: [maskSlot: mask, imageSlot: resource])
        }
        await #expect(throws: GPUError.self) { try await device.submitCommands([draw(mask)]) }
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff00ffff, 10]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: mask, region: .init(width: 1, height: 1)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 4), destination: image, region: .init(width: 1, height: 1)),
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            draw(image),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 4),
        ])
        let expected: [[UInt32]] = [[0xffff0000, 0xff00ffff], [15]]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
