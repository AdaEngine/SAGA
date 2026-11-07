import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let storageArrayShader = """
    @group(0) @binding(0) var source: texture_storage_2d_array<r32uint, read>;
    @group(0) @binding(1) var image: texture_storage_2d_array<r32uint, read_write>;
    @group(0) @binding(2) var<storage, read_write> result: array<u32, 10>;
    @compute @workgroup_size(1) fn run() {
        let xy = vec2i(0);
        let size = textureDimensions(image);
        let first = textureLoad(image, xy, 0i);
        let second = textureLoad(image, xy, 1u);
        let input = textureLoad(source, xy, 0u);
        result[0] = size.x;
        result[1] = size.y;
        result[2] = textureNumLayers(source);
        result[3] = textureNumLayers(image);
        result[4] = input.x;
        result[5] = first.x;
        result[6] = second.x;
        result[7] = textureLoad(image, xy, -1i).w;
        result[8] = textureLoad(image, xy, 0xffffffffu).w;
        result[9] = textureLoad(source, xy, 1u).w;
        textureStore(image, xy, 0i, first + input);
        textureStore(image, xy, 1u, second + vec4u(7u));
        textureStore(image, xy, -1i, vec4u(99u));
        textureStore(image, xy, 2u, vec4u(99u));
        textureStore(image, vec2u(0xffffffffu, 0u), 0u, vec4u(99u));
    }
    """

private let fragmentStorageArrayShader = """
    @group(1) @binding(3) var image: texture_storage_2d_array<rgba8unorm, write>;
    @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        if index == 0u { return vec4f(-1, -1, 0, 1); }
        if index == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    fn paint(value: vec4f) { textureStore(image, vec2i(0), 1u, value); }
    @fragment fn fs() -> @location(0) vec4f {
        paint(vec4f(1, 0, 1, 1));
        return vec4f(0, 1, 0, 1);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsStorageTextureArrayQueriesAndAccesses(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(storageArrayShader), to: format)
    #expect(shader.storageTextures.map(\.access) == [.load, [.load, .store]])
    #expect(shader.storageTextures.map(\.isArrayed) == [true, true])
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("texture_storage_2d_array<r32uint, read_write>"))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl:
        #expect(body.contains("texture2d_array<uint, access::read_write>"))
        #expect(body.contains(".get_array_size()"))
    case .hlsl:
        #expect(body.contains("RWTexture2DArray<uint>"))
        #expect(body.contains(".GetDimensions(") && body.contains("uint3("))
    case .glsl:
        #expect(body.contains("readonly uniform uimage2DArray"))
        #expect(body.contains("imageSize(") && body.contains("ivec3("))
    default: Issue.record("Unexpected format")
    }
    let fragment = try compiler.compileRenderStage(.wgsl(fragmentStorageArrayShader), entryPoint: "fs", to: format)
    #expect(fragment.storageTextures.map(\.isArrayed) == [true])
    if format == .wgsl, case .text(_, let source) = fragment.artifact {
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [
    "let value = textureLoad(image, vec2u(0));",
    "let value = textureLoad(image, vec2u(0), 1.0);",
    "let value = textureDimensions(image, 0u);",
    "textureStore(image, vec2u(0), vec4u(0));",
    "textureStore(image, vec2u(0), 1.0, vec4u(0));",
    "textureStore(source, vec2u(0), 0u, vec4u(0));",
])
func rejectsInvalidStorageArrayOperands(body: String) throws {
    let source = """
        @group(0) @binding(0) var source: texture_storage_2d_array<r32uint, read>;
        @group(0) @binding(1) var image: texture_storage_2d_array<r32uint, read_write>;
        @compute @workgroup_size(1) fn run() { \(body) }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid storage array access accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func storageTextureArrayUsesSelectedMipAndLayerRangeOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(storageArrayShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let texture = try await device.createTexture(.init(width: 4, height: 2, format: .r32Uint,
            usage: [.copySource, .copyDestination, .storageBinding], mipLevelCount: 2, arrayLayerCount: 4))
        let source = try await device.createTextureView(texture: texture,
            .init(baseMipLevel: 1, dimension: .d2Array, arrayLayerCount: 1))
        let image = try await device.createTextureView(texture: texture,
            .init(baseMipLevel: 1, baseArrayLayer: 1, arrayLayerCount: 2))
        let slice = try await device.createTextureView(texture: texture,
            .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 1))
        let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 40, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 40, usage: [.copyDestination, .mapRead]))
        func dispatch(source: ResourceID, image: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): source, .init(binding: 1): image])
        }
        await #expect(throws: GPUError.self) { try await device.submitCommands([dispatch(source: source, image: slice)]) }
        await #expect(throws: GPUError.self) { try await device.submitCommands([dispatch(source: image, image: image)]) }
        var commands: [BufferCommand] = [.write(buffer: upload, offset: 0, words: [6, 10, 20, 40])]
        for layer in UInt32(0)..<4 {
            commands.append(.copyBufferToTexture(source: upload, layout: .init(offset: UInt64(layer) * 4),
                destination: texture, region: .init(width: 1, height: 1, mipLevel: 1, arrayLayer: layer)))
        }
        commands += [dispatch(source: source, image: image),
                     .copy(source: output, destination: readback, byteCount: 40), .read(buffer: readback, byteCount: 40)]
        for layer in UInt32(0)..<4 {
            commands += [.copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256,
                                              mipLevel: 1, arrayLayer: layer), .read(buffer: readback, byteCount: 8)]
        }
        let compiler = IndependentShaderCompiler()
        let drawSource = ShaderInput.wgsl(fragmentStorageArrayShader)
        let draw = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(drawSource, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(drawSource, entryPoint: "fs", to: .msl), colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.storageBinding, .copySource], arrayLayerCount: 2))
        commands.append(.render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                                textures: [.init(group: 1, binding: 3): color]))
        for layer in UInt32(0)..<2 {
            commands += [.copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256, arrayLayer: layer),
                         .read(buffer: readback, byteCount: 4)]
        }
        let ticket = try await device.submitCommands(commands)
        let expected: [[UInt32]] = [[2, 1, 1, 2, 6, 10, 20, 0, 0, 0], [6, 0], [16, 0], [27, 0], [40, 0], [0], [0xffff00ff]]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
