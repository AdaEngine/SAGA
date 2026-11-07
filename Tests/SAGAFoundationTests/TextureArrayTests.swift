import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let arrayShader = """
    @group(0) @binding(0) var image: texture_2d_array<u32>;
    @group(0) @binding(1) var<storage, read_write> out: array<vec4u, 5>;
    @compute @workgroup_size(1) fn run() {
        out[0] = textureLoad(image, vec2u(0), 0u, 0u);
        out[1] = textureLoad(image, vec2i(1, 0), 1i, 0i);
        out[2] = textureLoad(image, vec2u(0), 2u, 0u);
        out[3] = textureLoad(image, vec2i(0), -1i, 0i) +
                 textureLoad(image, vec2u(0), 0xffffffffu, 0u) +
                 textureLoad(image, vec2u(0), 1u, 99u);
        out[4] = vec4u(textureDimensions(image, 0), textureNumLayers(image), textureNumLevels(image));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsTextureArrayLoadsAndQueries(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(arrayShader), to: format)
    #expect(shader.sampledTextures.first?.isArrayed == true)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("texture_2d_array<u32>") && body.contains("textureNumLayers("))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl:
        #expect(body.contains("texture2d_array<uint, access::sample>"))
        #expect(body.contains(".get_array_size()"))
    case .hlsl:
        #expect(body.contains("Texture2DArray<uint4>") && body.contains(".Load(int4("))
    case .glsl:
        #expect(body.contains("usampler2DArray") && body.contains("ivec3(ivec2("))
        #expect(body.contains("textureSize(") && body.contains(").z"))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    "textureLoad(image, vec2u(0), 0u)",
    "textureLoad(image, vec2u(0), 0.0, 0u)",
    "textureNumLayers(single)",
    "textureGather(0, image, state, vec2f(0.5), 0.0)",
])
func rejectsMismatchedTextureArrayOperations(call: String) throws {
    let source = """
        @group(0) @binding(0) var image: texture_2d_array<u32>;
        @group(0) @binding(1) var single: texture_2d<u32>;
        @group(0) @binding(2) var state: sampler;
        @compute @workgroup_size(1) fn run() { let value = \(call); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid texture array operation accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func loadsArrayLayersFromSelectedMipOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(arrayShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Uint,
            usage: [.copySource, .copyDestination, .textureBinding], mipLevelCount: 2, arrayLayerCount: 3))
        let view = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
        let single = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Uint, usage: .textureBinding))
        let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
        let out = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
        func load(_ texture: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 1): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): texture])
        }
        await #expect(throws: GPUError.validation("Invalid sampled texture binding")) {
            _ = try await device.submitCommands([load(single)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xf0800201, 0x04030201, 0x08070605, 0xffeeddcc]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: image,
                                 region: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 0)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: image,
                                 region: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 1)),
            load(view),
            .copy(source: out, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
            .copyTextureToTexture(source: image, destination: image,
                                  sourceRegion: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 0),
                                  destinationOrigin: .init(mipLevel: 1, arrayLayer: 2)),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1, arrayLayer: 2),
            .read(buffer: read, byteCount: 8),
        ])
        let expected: [UInt32] = [1, 2, 128, 240, 204, 221, 238, 255, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 1]
        #expect(try await ticket.wait() == [expected, [0xf0800201, 0x04030201]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

@Test func validatesTextureViewLayerRangesAndDimensions() throws {
    let array = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: .textureBinding, arrayLayerCount: 4)
    let layer = TextureViewDescriptor(dimension: .d2, baseArrayLayer: 2)
    #expect(try layer.arrayLayers(in: array) == 2..<3)
    #expect(try TextureViewDescriptor(baseArrayLayer: 2).arrayLayers(in: array) == 2..<4)
    #expect(throws: GPUError.self) {
        _ = try TextureViewDescriptor(dimension: .d2, arrayLayerCount: 2).arrayLayers(in: array)
    }
    #expect(throws: GPUError.self) {
        _ = try TextureViewDescriptor(baseArrayLayer: .max).arrayLayers(in: array)
    }
}

private let arrayViewShader = """
    @group(0) @binding(0) var source: texture_2d_array<u32>;
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4u {
        let x = select(0u, 1u, p.x > 1.0);
        let texel = textureLoad(source, vec2u(x, 0), 0u, 0u);
        return vec4u(texel.rgb, textureNumLayers(source));
    }
    """

#if canImport(Metal)
@Test(arguments: [TextureViewDimension.d2, .d2Array])
func rendersToOneArrayLayerWhileReadingAnotherOnMetal(dimension: TextureViewDimension) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(arrayViewShader)
        let pipeline = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl), colorFormat: .rgba8Uint)
        let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Uint,
            usage: [.renderAttachment, .textureBinding, .copySource, .copyDestination], mipLevelCount: 2, arrayLayerCount: 3))
        let target = try await device.createTextureView(texture: image,
            .init(baseMipLevel: 1, dimension: dimension, baseArrayLayer: 1, arrayLayerCount: 1))
        let sourceView = try await device.createTextureView(texture: image,
            .init(baseMipLevel: 1, dimension: .d2Array, baseArrayLayer: 2, arrayLayerCount: 1))
        let overlap = try await device.createTextureView(texture: image,
            .init(baseMipLevel: 1, dimension: .d2Array, baseArrayLayer: 1, arrayLayerCount: 1))
        let upload = try await device.createBuffer(.init(size: 8, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        func draw(_ sampled: ResourceID) -> BufferCommand {
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 0): sampled])
        }
        await #expect(throws: GPUError.validation("Render attachment cannot be bound as a sampled texture")) {
            _ = try await device.submitCommands([draw(overlap)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0x281e140a, 0x50463c32]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: image,
                                 region: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 2)),
            draw(sourceView),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1, arrayLayer: 1),
            .read(buffer: read, byteCount: 8),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1, arrayLayer: 2),
            .read(buffer: read, byteCount: 8),
        ])
        #expect(try await ticket.wait() == [[0x011e140a, 0x01463c32], [0x281e140a, 0x50463c32]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

private let layerStorageShader = """
    @group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
    @compute @workgroup_size(1) fn run() {
        textureStore(image, vec2u(0), vec4f(1, 0, 0, 1));
    }
    """

#if canImport(Metal)
@Test func writesStorageViewToSelectedArrayLayerOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compute = try await device.createComputePipeline(
            IndependentShaderCompiler().compileCompute(.wgsl(layerStorageShader), to: .msl))
        let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.storageBinding, .copySource], arrayLayerCount: 2))
        let slice = try await device.createTextureView(texture: image, .init(dimension: .d2, baseArrayLayer: 1))
        let array = try await device.createTextureView(texture: image,
            .init(dimension: .d2Array, baseArrayLayer: 1, arrayLayerCount: 1))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        func dispatch(_ texture: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: compute, bindings: [:], workgroups: .init(x: 1), textures: [.init(binding: 0): texture])
        }
        await #expect(throws: GPUError.self) {
            _ = try await device.submitCommands([dispatch(array)])
        }
        let ticket = try await device.submitCommands([
            dispatch(slice),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, arrayLayer: 1),
            .read(buffer: read, byteCount: 8),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, arrayLayer: 0),
            .read(buffer: read, byteCount: 8),
        ])
        #expect(try await ticket.wait() == [[0xff0000ff, 0], [0, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
