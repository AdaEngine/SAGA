import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private let packedFormats: [TextureFormat] = [.rgb10a2Unorm, .rgb10a2Uint, .rg11b10Ufloat]

@Test func packedColorUsageSeparatesCoreCopiesRenderabilityAndStorage() throws {
    let core = try DeviceCapabilities(maxBufferSize: 64)
    let extended = try DeviceCapabilities(maxBufferSize: 64, features: [.packedColorStorageTextures, .rg11b10UfloatRenderable])
    for format in packedFormats {
        try TextureDescriptor(width: 1, height: 1, format: format, usage: [.textureBinding, .copyDestination, .copySource])
            .validate(against: core)
        let color = TextureDescriptor(width: 1, height: 1, format: format, usage: .renderAttachment, sampleCount: 4)
        if format == .rg11b10Ufloat {
            #expect(throws: GPUError.validation("Feature not advertised")) { try color.validate(against: core) }
        } else {
            try color.validate(against: core)
        }
        try color.validate(against: extended)
        let storage = TextureDescriptor(width: 1, height: 1, format: format, usage: .storageBinding)
        #expect(throws: GPUError.validation("Feature not advertised")) { try storage.validate(against: core) }
        try storage.validate(against: extended)
    }
}

private let packedColorShader = """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn normalized() -> @location(0) vec4f { return vec4f(0, 0, 1, 2.0 / 3.0); }
    @fragment fn integer() -> @location(0) vec4u { return vec4u(1023, 512, 3, 2); }
    @fragment fn hdr() -> @location(0) vec3f { return vec3f(1, 2, 4); }
    """

private let packedSampleShader = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var state: sampler;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() { out = textureSampleLevel(image, state, vec2f(0.5), 0.0); }
    """

private let packedWriteShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var normalized: texture_storage_2d<rgb10a2unorm, write>;
    @group(0) @binding(1) var integer: texture_storage_2d<rgb10a2uint, write>;
    @group(0) @binding(2) var hdr: texture_storage_2d<rg11b10ufloat, write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(normalized, p, vec4f(1, 0, 1, 2.0 / 3.0));
        textureStore(integer, p, vec4u(1023, 512, 3, 2));
        textureStore(hdr, p, vec4f(1, 2, 4, 99));
    }
    """

private let packedReadShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var normalized: texture_storage_2d<rgb10a2unorm, read>;
    @group(0) @binding(1) var integer: texture_storage_2d<rgb10a2uint, read>;
    @group(0) @binding(2) var hdr: texture_storage_2d<rg11b10ufloat, read>;
    @group(0) @binding(3) var<storage, read_write> out: array<vec4u, 3>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        out[0] = bitcast<vec4u>(textureLoad(normalized, p));
        out[1] = textureLoad(integer, p);
        out[2] = bitcast<vec4u>(textureLoad(hdr, p));
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func packedStorageAndThreeComponentColorTranslate(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let hdr = try compiler.compileRenderStage(.wgsl(packedColorShader), entryPoint: "hdr", to: format)
    #expect(hdr.colorOutputComponentCounts == [0: 3])
    for (source, access) in [(packedWriteShader, IRStorageAccess.store), (packedReadShader, .load)] {
        let shader = try compiler.compileCompute(.wgsl(source), to: format)
        #expect(shader.storageTextures.map(\.format) == [.rgb10a2Unorm, .rgb10a2Uint, .rg11b10Ufloat])
        #expect(shader.storageTextures.allSatisfy { $0.access == access })
        #expect(shader.storageTextureFeatures == (access == .load ? [.packedColorStorageTextures, .storageTextureRead] : [.packedColorStorageTextures]))
        guard case .text(_, let code) = shader.artifact else { Issue.record("Expected shader source"); return }
        if format == .hlsl { #expect(code.contains("RWTexture2D<float4>")) }
        if format == .glsl { #expect(code.contains("layout(r11f_g11f_b10f,") && code.contains("layout(rgb10_a2ui,")) }
        if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code)) }
    }
}

#if canImport(Metal)
@Test func rendersResolvesAndSamplesPackedColorTexturesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.rg11b10UfloatRenderable])
    let words = try await packedColorDiagnostic(device)
    #expect(words == [0xbff003ff, 0x803803ff, 0x882003d0] + [Float(0), 0, 0, 1, 1, 2, 4, 1].map(\.bitPattern))
    await device.close()
}

private func packedColorDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(packedColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let additive = RenderBlendComponent(sourceFactor: .one, destinationFactor: .one)
    let blend = RenderColorState(blend: .init(color: additive, alpha: additive))
    var commands: [BufferCommand] = []
    for (format, entry) in zip(packedFormats, ["normalized", "integer", "hdr"]) {
        if format == .rg11b10Ufloat && !device.capabilities.features.contains(.rg11b10UfloatRenderable) { continue }
        let isInteger = format == .rgb10a2Uint
        let sampleCount: UInt32 = isInteger ? 1 : 4
        let fragment = try compiler.compileRenderStage(source, entryPoint: entry, to: .msl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: format,
                                                             sampleCount: sampleCount, colorState: isInteger ? .init() : blend)
        let output = try await device.createTexture(.init(width: 1, height: 1, format: format, usage: [.renderAttachment, .copySource]))
        let target = isInteger ? output : try await device.createTexture(.init(
            width: 1, height: 1, format: format, usage: .renderAttachment, sampleCount: sampleCount))
        let clear = isInteger ? .transparentBlack :
            RenderClearColor(red: format == .rgb10a2Unorm ? 1 : 0.25, green: 0, blue: 0, alpha: 0)
        commands += [
            .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                    resolve: isInteger ? nil : .init(texture: output)),
            .copyTextureToBuffer(source: output, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]
    }
    let hdr = try await device.createTexture(.init(width: 1, height: 1, format: .rg11b10Ufloat,
                                                     usage: [.copyDestination, .textureBinding]))
    let color = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let sampler = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(packedSampleShader), to: .msl))
    let capture: [BufferCommand] = [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): color], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): hdr], samplers: [.init(binding: 1): sampler]),
        .copy(source: color, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]
    commands += capture + [
        .writeTexture(destination: hdr, data: [0xc0, 0x03, 0x20, 0x88], region: .init(width: 1, height: 1)),
    ] + capture
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

@Test func storesAndReadsPackedColorTexelsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.packedColorStorageTextures])
    let words = try await packedStorageDiagnostic(device)
    #expect(words == [0xbff003ff, 0x803803ff, 0x882003c0] +
        [Float(1), 0, 1, 2.0 / 3.0].map(\.bitPattern) + [1023, 512, 3, 2] + [Float(1), 2, 4, 1].map(\.bitPattern))
    await device.close()
}

private func packedStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let write = try await device.createComputePipeline(compiler.compileCompute(.wgsl(packedWriteShader), to: .msl))
    let read = try await device.createComputePipeline(compiler.compileCompute(.wgsl(packedReadShader), to: .msl))
    var images: [ComputeBindingSlot: ResourceID] = [:]
    for (index, format) in packedFormats.enumerated() {
        images[.init(binding: UInt32(index))] = try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                                        usage: [.storageBinding, .copySource]))
    }
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
    var commands: [BufferCommand] = [.dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: images)]
    for index in packedFormats.indices {
        let image = images[.init(binding: UInt32(index))]!
        commands += [.copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
                     .read(buffer: readback, byteCount: 4)]
    }
    commands += [
        .dispatchCompute(pipeline: read, bindings: [.init(binding: 3): output], workgroups: .init(x: 1), textures: images),
        .copy(source: output, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}
#endif
