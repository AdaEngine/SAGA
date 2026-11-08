import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

@Test func srgbCopiesAreCompatibleOnlyWithTheMatchingLinearFormat() throws {
    let ids = (1...4).map { ResourceID(device: 1, serial: UInt64($0)) }
    let formats: [TextureFormat] = [.rgba8Unorm, .rgba8UnormSrgb, .bgra8Unorm, .bgra8UnormSrgb]
    let descriptors = Dictionary(uniqueKeysWithValues: zip(ids, formats.map {
        TextureDescriptor(width: 1, height: 1, format: $0, usage: [.copySource, .copyDestination])
    }))
    for (source, destination) in [(0, 1), (1, 0), (2, 3), (3, 2)] {
        try BufferCommand.copyTextureToTexture(source: ids[source], destination: ids[destination],
                                               sourceRegion: .init(width: 1, height: 1))
            .validate(device: 1, descriptors: [:], textures: descriptors)
    }
    #expect(throws: GPUError.self) {
        try BufferCommand.copyTextureToTexture(source: ids[0], destination: ids[3], sourceRegion: .init(width: 1, height: 1))
            .validate(device: 1, descriptors: [:], textures: descriptors)
    }
    let capabilities = try DeviceCapabilities(maxBufferSize: 64)
    try TextureDescriptor(width: 2, height: 1, format: .rgb9e5Ufloat,
                          usage: [.textureBinding, .copySource, .copyDestination]).validate(against: capabilities)
    #expect(throws: GPUError.validation("Feature not advertised")) {
        try TextureDescriptor(width: 2, height: 1, format: .rgb9e5Ufloat, usage: .renderAttachment)
            .validate(against: capabilities)
    }
    #expect(throws: GPUError.validation("Invalid format for a storage texture")) {
        try TextureDescriptor(width: 2, height: 1, format: .rgb9e5Ufloat, usage: .storageBinding)
            .validate(against: capabilities)
    }
}

private let srgbReadShader = """
    @group(0) @binding(0) var rgba: texture_2d<f32>;
    @group(0) @binding(1) var bgra: texture_2d<f32>;
    @group(0) @binding(2) var linear: texture_2d<f32>;
    @group(0) @binding(3) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        let a = textureLoad(rgba, p, 0u);
        let b = textureLoad(bgra, p, 0u);
        let c = textureLoad(linear, p, 0u);
        out = vec4f(a.x, b.y, a.w, c.x);
    }
    """

private let srgbRenderShader = """
    struct Colors { @location(0) rgba: vec4f, @location(1) bgra: vec4f }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn srgb() -> Colors {
        let color = vec4f(0.25, 0.5, 0.75, 0.5);
        return Colors(color, color);
    }
    @fragment fn hdr() -> @location(0) vec3f { return vec3f(1.015625, 2.015625, 4.015625); }
    """

private let sharedExponentSampleShader = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var state: sampler;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() { out = textureSampleLevel(image, state, vec2f(0.5), 0.0); }
    """

#if canImport(Metal)
@Test func copiesDecodesAndResolvesSRGBColorFormatsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let words = try await srgbDiagnostic(device)
    #expect(Array(words.prefix(4)) == [0x80204080, 0x80204080, 0x80804020, 0x80804020])
    let decoded = words[4..<8].map(Float.init(bitPattern:))
    let expectedDecode: [Float] = [0.21586, 0.05127, 128.0 / 255.0, 128.0 / 255.0]
    for (actual, reference) in zip(decoded, expectedDecode) {
        #expect(abs(actual - reference) < 0.001)
    }
    for (word, expected) in zip(words.suffix(2), [[137, 188, 225, 128], [225, 188, 137, 128]]) {
        for (lane, value) in expected.enumerated() { #expect(abs(Int((word >> (8 * lane)) & 255) - value) <= 1) }
    }
    await device.close()
}

private func srgbDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    var images: [ResourceID] = []
    var commands: [BufferCommand] = []
    for (linear, srgb, bytes) in [(TextureFormat.rgba8Unorm, TextureFormat.rgba8UnormSrgb, [UInt8(128), 64, 32, 128]),
                                  (.bgra8Unorm, .bgra8UnormSrgb, [32, 64, 128, 128])] {
        let source = try await device.createTexture(.init(width: 1, height: 1, format: linear, usage: usage))
        let encoded = try await device.createTexture(.init(width: 1, height: 1, format: srgb, usage: usage))
        let copy = try await device.createTexture(.init(width: 1, height: 1, format: linear, usage: usage))
        images += [encoded, copy]
        let region = TextureCopyRegion(width: 1, height: 1)
        commands += [
            .writeTexture(destination: source, data: bytes, region: region),
            .copyTextureToTexture(source: source, destination: encoded, sourceRegion: region),
            .copyTextureToTexture(source: encoded, destination: copy, sourceRegion: region),
            .copyTextureToBuffer(source: encoded, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: copy, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]
    }
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(srgbReadShader), to: .msl))
    commands += [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 3): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): images[0], .init(binding: 1): images[2], .init(binding: 2): images[1]]),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]
    let source = ShaderInput.wgsl(srgbRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "srgb", to: .msl)
    let blend = RenderColorState(blend: .init())
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8UnormSrgb,
        sampleCount: 4, colorState: blend, additionalColorTargets: [1: .init(format: .bgra8UnormSrgb, colorState: blend)])
    let rgba = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8UnormSrgb, usage: .renderAttachment, sampleCount: 4))
    let bgra = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8UnormSrgb, usage: .renderAttachment, sampleCount: 4))
    let resolvedRGBA = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8UnormSrgb, usage: [.renderAttachment, .copySource]))
    let resolvedBGRA = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8UnormSrgb, usage: [.renderAttachment, .copySource]))
    commands += [
        .render(pipeline: render, target: rgba, clear: .transparentBlack, vertexCount: 3, resolve: .init(texture: resolvedRGBA),
                additionalColors: [1: .init(texture: bgra, loadAction: .clear(.transparentBlack), resolve: .init(texture: resolvedBGRA))]),
        .copyTextureToBuffer(source: resolvedRGBA, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: resolvedBGRA, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

@Test func copiesFiltersAndRendersSharedExponentTexturesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let words = try await sharedExponentDiagnostic(device)
    let expected = [Float(0), 0, 0, 1].map(\.bitPattern) + [0x94050241, 0xa5054221] +
        [Float(1.5390625), 6.0390625, 12.0390625, 1].map(\.bitPattern)
    let rendered: [UInt32] = device.capabilities.features.contains(.rgb9e5Renderable) ? [0x94050241] : []
    #expect(words == expected + rendered)
    await device.close()
}

private func sharedExponentDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let usage: TextureUsage = [.textureBinding, .copySource, .copyDestination]
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgb9e5Ufloat, usage: usage))
    let clone = try await device.createTexture(.init(width: 2, height: 1, format: .rgb9e5Ufloat, usage: usage))
    let data = [UInt32(0x94050241), 0xa5054221].withUnsafeBytes { Array($0) }
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let sampler = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(sharedExponentSampleShader), to: .msl))
    let sample: [BufferCommand] = [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): clone], samplers: [.init(binding: 1): sampler]),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]
    let region = TextureCopyRegion(width: 2, height: 1)
    var commands = sample + [
        .writeTexture(destination: image, data: data, region: region),
        .copyTextureToTexture(source: image, destination: clone, sourceRegion: region),
        .copyTextureToBuffer(source: clone, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
    ] + sample
    if device.capabilities.features.contains(.rgb9e5Renderable) {
        let source = ShaderInput.wgsl(srgbRenderShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "hdr", to: .msl)
        let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgb9e5Ufloat)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgb9e5Ufloat,
                                                          usage: [.renderAttachment, .copySource]))
        commands += [
            .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 3),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}
#endif
