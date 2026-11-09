import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvTextureInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var nearest: sampler;
    @group(0) @binding(2) var<storage, read_write> result: array<vec4u, 3>;

    @compute @workgroup_size(1) fn textureMip() {
        let uv = vec2f(0.75, 0.5);
        result[0] = bitcast<vec4u>(textureSampleLevel(image, nearest, uv, 1.0));
        result[1] = bitcast<vec4u>(textureLoad(image, vec2i(0), 1));
        result[2] = vec4u(textureDimensions(image, 1), textureNumLevels(image), textureDimensions(image).x);
    }
    """)

private func exportTextureModule(_ compiler: IndependentShaderCompiler, _ input: ShaderInput = spirvTextureInput) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvTextureCodeGeneratorExportsSampleFetchAndMipQueries(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportTextureModule(compiler)
    try #require(words.count > 5)
    var opcodes: Set<UInt32> = []
    var capabilities: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        if opcode == 17 && count == 2 { capabilities.insert(words[offset + 1]) }
        offset += count
    }
    // OpTypeImage, OpTypeSampler, OpTypeSampledImage, OpSampledImage, OpImageSampleExplicitLod,
    // OpImageFetch, OpImageQuerySizeLod, OpImageQueryLevels.
    #expect(opcodes.isSuperset(of: [25, 26, 27, 86, 88, 95, 103, 106]))
    #expect(capabilities.contains(50)) // ImageQuery

    let input = ShaderInput.spirv(words)
    let module = try compiler.parseAndValidate(input).module
    #expect(module.entryPoints.map(\.name) == ["textureMip"])
    let shader = try compiler.compileCompute(input, entryPoint: "textureMip", to: format)
    let texture = try #require(shader.sampledTextures.first)
    let sampler = IRResourceBinding(group: 0, binding: 1)
    #expect(shader.sampledTextures.count == 1)
    #expect(texture.binding == .init(group: 0, binding: 0) && texture.dimension == .d2)
    #expect(texture.sampleKind == .float && texture.samplingSamplers == [sampler])
    #expect(shader.samplerBindings.map(\.binding) == [sampler])
    #expect(shader.storageBindings.map(\.minimumByteSize) == [48])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
        _ = try compiler.compileWebGPUCompute(input, entryPoint: "textureMip")
    }
}

@Test(arguments: [false, true])
func spirvTextureCodeGeneratorRoundTripsIntegerRawDepthLod(_ dynamic: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let declaration = dynamic ? "@group(0) @binding(3) var<storage, read> level: i32;" : ""
    let level = dynamic ? "level" : "1i"
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var depth: texture_depth_2d;
        @group(0) @binding(1) var nearest: sampler;
        @group(0) @binding(2) var<storage, read_write> result: f32;
        \(declaration)
        @compute @workgroup_size(1) fn sampleDepth() {
            result = textureSampleLevel(depth, nearest, vec2f(0.5), \(level));
        }
        """)
    let input = ShaderInput.spirv(try exportTextureModule(compiler, source))
    let module = try compiler.parseAndValidate(input).module
    let function = try #require(module.functions.values.first { function in
        function.expressions.values.contains { if case .imageSample = $0.kind { true } else { false } }
    })
    let levels = function.expressions.values.compactMap { expression -> IRSampleLevel? in
        if case .imageSample(let sample) = expression.kind { return sample.level }
        return nil
    }
    guard case .exact(let lod) = try #require(levels.first) else {
        Issue.record("Raw depth sample lost its explicit mip level")
        return
    }
    let lodType = try module.types.value(for: function.expressions.value(for: lod).type).kind
    #expect(lodType == .scalar(.i32) || lodType == .scalar(.u32))
    let shader = try compiler.compileCompute(input, entryPoint: "sampleDepth", to: .wgsl)
    #expect(shader.sampledTextures.first?.isDepth == true)
    #expect(shader.samplerBindings.first?.requiresNonFiltering == true)
    guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.wgsl(text))
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvTextureCodeGeneratorRoundTripsStorageImageAccess(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var source: texture_storage_2d<r32uint, read>;
        @group(0) @binding(1) var destination: texture_storage_2d<r32uint, write>;
        @group(0) @binding(2) var scratch: texture_storage_2d<r32uint, read_write>;
        @compute @workgroup_size(1) fn run() {
            let p = vec2i(1, 0);
            let size = textureDimensions(destination);
            let red = textureLoad(source, p).x + textureLoad(scratch, p).x + 100u * size.x + size.y;
            textureStore(destination, p, vec4u(red, 0u, 0u, 0u));
            textureStore(scratch, p, vec4u(red + 1u, 0u, 0u, 0u));
        }
        """)
    let input = ShaderInput.spirv(try exportTextureModule(compiler, source))
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.storageTextures.map(\.binding.binding) == [0, 1, 2])
    #expect(shader.storageTextures.map(\.format) == [.r32Uint, .r32Uint, .r32Uint])
    #expect(shader.storageTextures.map(\.access) == [.load, .store, [.load, .store]])
}

#if canImport(Metal)
@Test func spirvTextureCodeGeneratorPreservesSelectedMipAndTexelsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportTextureModule(compiler)
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "textureMip", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let texture = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: [.textureBinding, .copyDestination], mipLevelCount: 2))
        let sampler = try await device.createSampler(.init(mipmapFilter: .nearest))
        let upload = try await device.createBuffer(.init(size: 280, usage: [.copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
        let red = Array(repeating: UInt32(0xff0000ff), count: 4)
        let snapshots = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: red),
            .write(buffer: upload, offset: 256, words: red + [0xffff0000, 0xff00ff00]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: texture,
                                 region: .init(width: 4, height: 2)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 272), destination: texture,
                                 region: .init(width: 2, height: 1, mipLevel: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): texture], samplers: [.init(binding: 1): sampler]),
            .copy(source: output, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
        ]).wait()
        let colors: [Float] = [0, 1, 0, 1, 0, 0, 1, 1]
        #expect(snapshots == [colors.map(\.bitPattern) + [2, 1, 2, 4]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
