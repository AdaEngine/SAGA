import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let textureAtomicSource = """
    override increment = 1u;
    @group(0) @binding(0) var positive: texture_storage_2d<r32uint, atomic>;
    @group(0) @binding(1) var negative: texture_storage_2d<r32sint, atomic>;
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        textureAtomicAdd(positive, vec2i(0), increment);
        textureAtomicAdd(negative, vec2u(0), -2);
        if lane == 0u {
            let pixel = vec2i(1, 0);
            textureAtomicMax(positive, pixel, 5u);
            textureAtomicMin(positive, pixel, 6u);
            textureAtomicAnd(positive, pixel, 7u);
            textureAtomicOr(positive, pixel, 8u);
            textureAtomicXor(positive, pixel, 1u);
            textureAtomicMax(negative, pixel, -3);
            textureAtomicMin(negative, pixel, -4);
            textureAtomicAnd(negative, pixel, -2);
            textureAtomicOr(negative, pixel, 1);
            textureAtomicXor(negative, pixel, 1);
        }
        textureAtomicAdd(positive, vec2i(-1, 0), 99u);
        textureAtomicAdd(negative, vec2u(99, 0), 99i);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func textureAtomicWritersPreserveSignedOperationsAndAccess(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(textureAtomicSource), to: format)
    #expect(shader.storageTextures.count == 2)
    #expect(shader.storageTextures.allSatisfy { $0.access == [.load, .store, .atomic] })
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    switch format {
    case .msl: #expect(source.contains(".atomic_fetch_xor("))
    case .hlsl: #expect(source.contains("InterlockedXor("))
    case .glsl: #expect(source.contains("imageAtomicXor("))
    case .wgsl:
        #expect(source.contains("r32sint, atomic>"))
        #expect(source.contains("textureAtomicXor("))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.textureAtomic))
    default: Issue.record("Unexpected output format")
    }
}

@Test(arguments: [
    ("texture_storage_2d<r32uint, read_write>", "textureAtomicAdd(image, vec2u(0), 1u)"),
    ("texture_storage_2d<rgba8unorm, atomic>", "textureAtomicAdd(image, vec2u(0), 1u)"),
    ("texture_storage_2d<r32uint, atomic>", "textureAtomicAdd(image, vec2u(0), 1i)"),
    ("texture_storage_2d<r32uint, atomic>", "textureAtomicAdd(image, vec3u(0), 1u)"),
])
func textureAtomicsRejectInvalidAccessAndArguments(type: String, statement: String) {
    let source = "@group(0) @binding(0) var image: \(type); @compute @workgroup_size(1) fn run() { \(statement); }"
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.textureAtomic))
    }
}

@Test func textureAtomicsRequireCapabilityAndNativeWGSLTarget() {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(textureAtomicSource)) }
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(.wgsl(textureAtomicSource)) }
}

#if canImport(Metal)
@Test func textureAtomicsExecuteThroughMetalDevice() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.textureAtomic) else { return }
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(textureAtomicSource), to: .msl)
    let pipeline = try await device.createComputePipeline(shader)
    let usage: TextureUsage = [.storageAtomic, .copyDestination, .copySource]
    let positive = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint, usage: usage))
    let negative = try await device.createTexture(.init(width: 2, height: 1, format: .r32Sint, usage: usage))
    let ordinary = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint, usage: .storageBinding))
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
            textures: [.init(binding: 0): ordinary, .init(binding: 1): negative])])
    }
    let readback = try await device.createBuffer(.init(size: 8, usage: [.mapRead, .copyDestination]))
    let region = TextureCopyRegion(width: 2, height: 1)
    let words = try await device.submitCommands([
        .writeTexture(destination: positive, data: [10, 0, 0, 0, 3, 0, 0, 0], region: region),
        .writeTexture(destination: negative, data: [251, 255, 255, 255, 254, 255, 255, 255], region: region),
        .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): positive, .init(binding: 1): negative]),
        .copyTextureToBuffer(source: positive, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        .copyTextureToBuffer(source: negative, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
    ]).wait()
    #expect(words == [[14, 12], [UInt32(bitPattern: -13), UInt32(bitPattern: -4)]])
    await device.close()
}
#endif

private let textureAtomicRenderSource = """
    @group(0) @binding(0) var counts: texture_storage_2d_array<r32uint, atomic>;
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        let xy = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return vec4f(xy[i], 0, 1);
    }
    fn bump() {
        textureAtomicAdd(counts, vec2i(0), 1u, 1u);
        textureAtomicAdd(counts, vec2i(0), 2u, 99u);
    }
    @fragment fn fs(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
        bump();
        return vec4f(0, 0, 0, 1);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func textureAtomicRenderReflectionPreservesArrayLayers(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(textureAtomicRenderSource)
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: format)
    #expect(vertex.storageTextures.isEmpty)
    #expect(fragment.storageTextures.first?.isArrayed == true)
    #expect(fragment.storageTextures.first?.access == [.load, .store, .atomic])
    if format == .msl { #expect(fragment.metalTextureAtomicGuardBufferIndex != nil) }
}

#if canImport(Metal)
@Test func textureAtomicFragmentsUpdateSelectedMetalArrayLayer() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.textureAtomic) else { return }
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(textureAtomicRenderSource)
    let pipeline = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(input, entryPoint: "vs", to: .msl),
        fragment: compiler.compileRenderStage(input, entryPoint: "fs", to: .msl), colorFormat: .rgba8Unorm)
    let counts = try await device.createTexture(.init(width: 1, height: 1, format: .r32Uint,
        usage: [.storageAtomic, .copyDestination, .copySource], arrayLayerCount: 2))
    let color = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    let words = try await device.submitCommands([
        .writeTexture(destination: counts, data: [17, 0, 0, 0], region: .init(width: 1, height: 1)),
        .writeTexture(destination: counts, data: [3, 0, 0, 0], region: .init(width: 1, height: 1, arrayLayer: 1)),
        .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): counts]),
        .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): counts]),
        .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): counts]),
        .copyTextureToBuffer(source: counts, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: counts, destination: readback, bytesPerRow: 256, arrayLayer: 1),
        .read(buffer: readback, byteCount: 4),
    ]).wait()
    #expect(words == [[17], [27]])
    await device.close()
}
#endif
