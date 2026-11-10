import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let volumeShader = """
    @group(0) @binding(0) var volume: texture_3d<f32>;
    @group(0) @binding(1) var nearest: sampler;
    @group(0) @binding(2) var<storage, read_write> result: array<vec4f, 10>;
    @compute @workgroup_size(1) fn run() {
        result[0] = textureLoad(volume, vec3i(0, 0, 0), 0i);
        result[1] = textureLoad(volume, vec3u(0, 0, 1), 0u);
        result[2] = textureLoad(volume, vec3i(0, 0, 2), 0i);
        result[3] = textureLoad(volume, vec3i(0, 0, -1), 0i);
        result[4] = textureLoad(volume, vec3u(0, 0, 0xffffffffu), 0i);
        result[5] = textureSampleLevel(volume, nearest, vec3f(0.25, 0.5, 0.5), 0.0);
        result[6] = textureSampleGrad(volume, nearest, vec3f(0.25, 0.5, 0.16666667), vec3f(0), vec3f(0), vec3i(0, 0, 1));
        let size = textureDimensions(volume, 0u);
        result[7] = vec4f(f32(size.x), f32(size.y), f32(size.z), f32(textureNumLevels(volume)));
        let invalid = textureDimensions(volume, 1u);
        result[8] = vec4f(f32(invalid.x), f32(invalid.y), f32(invalid.z), textureLoad(volume, vec3u(0), -1i).a);
        result[9] = textureSampleLevel(volume, nearest, vec3f(0.25, 0.5, 0.5), 0.0, vec3i(0, 0, -1));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsVolumeTextureSamplingAndRobustLoads(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(volumeShader), to: format)
    #expect(shader.sampledTextures.first?.dimension == .d3)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("texture_3d<f32>"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains("texture3d<float, access::sample>"))
        #expect(source.contains("gradient3d(") && source.contains(".get_depth("))
    case .hlsl:
        #expect(source.contains("Texture3D<float4>"))
        #expect(source.contains(".Load(int4(") && source.contains(".SampleGrad("))
    case .glsl:
        #expect(source.contains("sampler3D"))
        #expect(source.contains("texelFetch(") && source.contains("textureGradOffset("))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    "textureLoad(volume, vec2u(0), 0u)",
    "textureGather(0, volume, nearest, vec3f(0.5))",
    "textureSampleLevel(volume, nearest, vec3f(0.5), 0.0, vec2i(0))",
    "textureNumLayers(volume)",
])
func rejectsInvalidVolumeTextureOperations(expression: String) throws {
    let source = """
        @group(0) @binding(0) var volume: texture_3d<f32>;
        @group(0) @binding(1) var nearest: sampler;
        @compute @workgroup_size(1) fn run() { let value = \(expression); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid volume texture operation accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func copiesAndSamplesVolumeMipPlanesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(volumeShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let texture = TextureDescriptor(width: 4, height: 2, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination, .textureBinding], mipLevelCount: 2, dimension: .d3, depth: 6)
        let source = try await device.createTexture(texture)
        let volume = try await device.createTexture(texture)
        let view = try await device.createTextureView(texture: volume, .init(baseMipLevel: 1))
        await #expect(throws: GPUError.validation("Texture view dimension does not match the texture")) {
            try await device.createTextureView(texture: volume, .init(dimension: .d2))
        }
        let nearest = try await device.createSampler(.init())
        let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
        let out = try await device.createBuffer(.init(size: 160, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 520, usage: [.copyDestination, .mapRead]))
        let region = TextureCopyRegion(width: 2, height: 1, mipLevel: 1, depth: 2)
        await #expect(throws: GPUError.validation("Invalid texture copy source or aspect")) {
            try await device.submitCommands([.copyTextureToBuffer(source: volume, destination: read,
                bytesPerRow: 256, mipLevel: 1, z: 2, depth: 2)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff0000ff]),
            .write(buffer: upload, offset: 256, words: [0xff00ff00, 0xff00ff00]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: source, region: region),
            .copyTextureToTexture(source: source, destination: volume, sourceRegion: region, destinationOrigin: .init(mipLevel: 1)),
            .copyTextureToTexture(source: source, destination: volume,
                sourceRegion: .init(width: 1, height: 1, mipLevel: 1, z: 1), destinationOrigin: .init(x: 1, mipLevel: 1, z: 2)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): view], samplers: [.init(binding: 1): nearest]),
            .copy(source: out, destination: read, byteCount: 160), .read(buffer: read, byteCount: 160),
            .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, mipLevel: 1, depth: 3),
            .read(buffer: read, byteCount: 520),
        ])
        let snapshots = try await ticket.wait()
        let expected: [Float] = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 3, 1, 0, 0, 0, 0, 1, 0, 0, 1]
        #expect(snapshots[0] == expected.map(\.bitPattern))
        #expect([0, 1, 64, 65, 128, 129].map { snapshots[1][$0] } ==
                [0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00, 0, 0xff00ff00])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

#if canImport(Metal)
@Test func copiesPartialSingleVolumePlaneOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination], dimension: .d3, depth: 2))
        let upload = try await device.createBuffer(.init(size: 260, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff0000ff]),
            .write(buffer: upload, offset: 256, words: [0xff00ff00]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: texture,
                                 region: .init(width: 1, height: 2, z: 1)),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, z: 1),
            .read(buffer: read, byteCount: 264),
        ])
        let snapshots = try await ticket.wait()
        #expect(snapshots[0][0] == 0xff0000ff)
        #expect(snapshots[0][64] == 0xff00ff00)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
