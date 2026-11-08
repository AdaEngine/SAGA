import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private let integer16Formats: [TextureFormat] = [.r16Uint, .rg16Uint, .rgba16Uint, .r16Sint, .rg16Sint, .rgba16Sint]

@Test(arguments: integer16Formats)
func integer16ColorFormatsAreCoreAndNarrowStorageIsOptional(format: TextureFormat) throws {
    let core = try DeviceCapabilities(maxBufferSize: 128)
    let tier1 = try DeviceCapabilities(maxBufferSize: 128, features: [.narrowInteger16StorageTextures])
    try TextureDescriptor(width: 1, height: 1, format: format,
                          usage: [.textureBinding, .renderAttachment], sampleCount: 4).validate(against: core)
    let storage = TextureDescriptor(width: 1, height: 1, format: format, usage: .storageBinding)
    if format == .rgba16Uint || format == .rgba16Sint {
        try storage.validate(against: core)
    } else {
        #expect(throws: GPUError.validation("Feature not advertised")) { try storage.validate(against: core) }
        try storage.validate(against: tier1)
    }
}

private let integer16ColorShader = """
    struct UnsignedOut { @location(0) r: u32, @location(1) rg: vec2u, @location(2) rgba: vec4u }
    struct SignedOut { @location(0) r: i32, @location(1) rg: vec2i, @location(2) rgba: vec4i }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn unsignedColor() -> UnsignedOut {
        return UnsignedOut(60000u, vec2u(65535, 300), vec4u(1, 257, 65534, 1234));
    }
    @fragment fn signedColor() -> SignedOut {
        return SignedOut(-20000i, vec2i(-32768, 300), vec4i(-1, 0, 32767, -1234));
    }
    """

private func integer16LoadShader(multisampled: Bool) -> String {
    let texture = multisampled ? "texture_multisampled_2d" : "texture_2d"
    let index = multisampled ? "3u" : "0u"
    return """
        @group(0) @binding(0) var ur: \(texture)<u32>;
        @group(0) @binding(1) var urg: \(texture)<u32>;
        @group(0) @binding(2) var urgba: \(texture)<u32>;
        @group(0) @binding(3) var sr: \(texture)<i32>;
        @group(0) @binding(4) var srg: \(texture)<i32>;
        @group(0) @binding(5) var srgba: \(texture)<i32>;
        @group(0) @binding(6) var<storage, read_write> out: array<vec4u, 6>;
        @compute @workgroup_size(1) fn run() {
            let p = vec2u(0);
            out[0] = textureLoad(ur, p, \(index));
            out[1] = textureLoad(urg, p, \(index));
            out[2] = textureLoad(urgba, p, \(index));
            out[3] = bitcast<vec4u>(textureLoad(sr, p, \(index)));
            out[4] = bitcast<vec4u>(textureLoad(srg, p, \(index)));
            out[5] = bitcast<vec4u>(textureLoad(srgba, p, \(index)));
        }
        """
}

private let integer16WriteShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var ur: texture_storage_2d<r16uint, write>;
    @group(0) @binding(1) var urg: texture_storage_2d<rg16uint, write>;
    @group(0) @binding(2) var sr: texture_storage_2d<r16sint, write>;
    @group(0) @binding(3) var srg: texture_storage_2d<rg16sint, write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(ur, p, vec4u(60000, 7, 8, 9));
        textureStore(urg, p, vec4u(65535, 300, 8, 9));
        textureStore(sr, p, vec4i(-20000, 7, 8, 9));
        textureStore(srg, p, vec4i(-32768, 300, 8, 9));
    }
    """

private let integer16ReadShader = """
    requires texture_formats_tier1, readonly_and_readwrite_storage_textures;
    @group(0) @binding(0) var ur: texture_storage_2d<r16uint, read>;
    @group(0) @binding(1) var urg: texture_storage_2d<rg16uint, read>;
    @group(0) @binding(2) var sr: texture_storage_2d<r16sint, read>;
    @group(0) @binding(3) var srg: texture_storage_2d<rg16sint, read>;
    @group(0) @binding(4) var<storage, read_write> out: array<vec4u, 4>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        out[0] = textureLoad(ur, p);
        out[1] = textureLoad(urg, p);
        out[2] = bitcast<vec4u>(textureLoad(sr, p));
        out[3] = bitcast<vec4u>(textureLoad(srg, p));
    }
    """

private let integer16UpdateShader = """
    requires texture_formats_tier1, readonly_and_readwrite_storage_textures;
    @group(0) @binding(0) var ur: texture_storage_2d<r16uint, read_write>;
    @group(0) @binding(1) var sr: texture_storage_2d<r16sint, read_write>;
    @group(0) @binding(2) var urgba: texture_storage_2d<rgba16uint, read_write>;
    @group(0) @binding(3) var srgba: texture_storage_2d<rgba16sint, read_write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(ur, p, textureLoad(ur, p) + vec4u(1000));
        textureStore(sr, p, textureLoad(sr, p) - vec4i(1000));
        textureStore(urgba, p, textureLoad(urgba, p) + vec4u(1, 2, 3, 4));
        textureStore(srgba, p, textureLoad(srgba, p) - vec4i(1, 2, 3, 4));
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func integer16StorageFormatsReflectTypesAndAccessGates(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let write = try compiler.compileCompute(.wgsl(integer16WriteShader), to: format)
    let read = try compiler.compileCompute(.wgsl(integer16ReadShader), to: format)
    let update = try compiler.compileCompute(.wgsl(integer16UpdateShader), to: format)
    #expect(write.storageTextures.map(\.format) == [.r16Uint, .rg16Uint, .r16Sint, .rg16Sint])
    #expect(write.storageTextureFeatures == [.narrowInteger16StorageTextures])
    #expect(read.storageTextureFeatures == [.narrowInteger16StorageTextures, .storageTextureRead])
    #expect(update.storageTextures.map(\.format) == [.r16Uint, .r16Sint, .rgba16Uint, .rgba16Sint])
    #expect(update.storageTextureFeatures == [.narrowInteger16StorageTextures, .integer16StorageReadWrite, .storageTextureReadWrite])
    if format == .wgsl {
        for shader in [write, read, update] {
            guard case .text(_, let code) = shader.artifact else { Issue.record("Expected shader source"); return }
            _ = try compiler.parseAndValidate(.wgsl(code))
        }
    }
}

#if canImport(Metal)
@Test func rendersAndSamplesAllInteger16ColorFormatsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let words = try await integer16ColorDiagnostic(device)
    let expected: [Int32] = [
        60000, 0, 0, 1, 65535, 300, 0, 1, 1, 257, 65534, 1234,
        -20000, 0, 0, 1, -32768, 300, 0, 1, -1, 0, 32767, -1234,
    ]
    #expect(words == [0xea600000, 0xb1e00000] + expected.map(UInt32.init(bitPattern:)))
    let multisampled = try await integer16ColorDiagnostic(device, sampleCount: 4)
    #expect(multisampled == expected.map(UInt32.init(bitPattern:)))
    await device.close()
}

private func integer16ColorDiagnostic(_ device: Device, sampleCount: UInt32 = 1) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(integer16ColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(integer16LoadShader(multisampled: sampleCount > 1)), to: .msl))
    let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 96, usage: [.mapRead, .copyDestination]))
    var usage: TextureUsage = [.renderAttachment, .textureBinding]
    if sampleCount == 1 { usage.insert(.copySource) }
    var textures: [ResourceID] = []
    for format in integer16Formats {
        textures.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                               usage: usage, sampleCount: sampleCount)))
    }
    var commands: [BufferCommand] = []
    for (base, entry) in [(0, "unsignedColor"), (3, "signedColor")] {
        let fragment = try compiler.compileRenderStage(source, entryPoint: entry, to: .msl)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: integer16Formats[base], sampleCount: sampleCount,
            additionalColorTargets: [1: .init(format: integer16Formats[base + 1]),
                                     2: .init(format: integer16Formats[base + 2])])
        commands.append(.render(pipeline: pipeline, target: textures[base], clear: .transparentBlack, vertexCount: 3,
                                additionalColors: [1: .init(texture: textures[base + 1]), 2: .init(texture: textures[base + 2])]))
        if sampleCount == 1 {
            commands += [
                .copyTextureToBuffer(source: textures[base], destination: readback, bytesPerRow: 256, destinationOffset: 2),
                .read(buffer: readback, byteCount: 4),
            ]
        }
    }
    let bindings = Dictionary(uniqueKeysWithValues: textures.enumerated().map { (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element) })
    commands += [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 6): output], workgroups: .init(x: 1), textures: bindings),
        .copy(source: output, destination: readback, byteCount: 96), .read(buffer: readback, byteCount: 96),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

@Test func storesReadsAndUpdatesInteger16TexturesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.narrowInteger16StorageTextures])
    let words = try await integer16StorageDiagnostic(device)
    let expected: [Int32] = [60000, 0, 0, 1, 65535, 300, 0, 1, -20000, 0, 0, 1, -32768, 300, 0, 1]
    let updated: [UInt32] = [0xee480000, 0xadf80000, 0x00020001, 0x00040003, 0xfffeffff, 0xfffcfffd]
    #expect(words == expected.map(UInt32.init(bitPattern:)) +
        (device.capabilities.features.contains(.integer16StorageReadWrite) ? updated : []))
    await device.close()
}

private func integer16StorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let write = try await device.createComputePipeline(compiler.compileCompute(.wgsl(integer16WriteShader), to: .msl))
    let read = try await device.createComputePipeline(compiler.compileCompute(.wgsl(integer16ReadShader), to: .msl))
    var images: [ResourceID] = []
    for format: TextureFormat in [.r16Uint, .rg16Uint, .r16Sint, .rg16Sint, .rgba16Uint, .rgba16Sint] {
        images.append(try await device.createTexture(.init(width: 1, height: 1, format: format, usage: [.storageBinding, .copySource])))
    }
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
    let bindings = Dictionary(uniqueKeysWithValues: images.prefix(4).enumerated().map {
        (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element)
    })
    var commands: [BufferCommand] = [
        .dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: bindings),
        .dispatchCompute(pipeline: read, bindings: [.init(binding: 4): output], workgroups: .init(x: 1), textures: bindings),
        .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
    ]
    if device.capabilities.features.contains(.integer16StorageReadWrite) {
        let update = try await device.createComputePipeline(compiler.compileCompute(.wgsl(integer16UpdateShader), to: .msl))
        commands += [
            .dispatchCompute(pipeline: update, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): images[0], .init(binding: 1): images[2],
                                        .init(binding: 2): images[4], .init(binding: 3): images[5]]),
            .write(buffer: readback, offset: 0, words: [0]),
        ]
        for index in [0, 2, 4, 5] {
            commands += [
                .copyTextureToBuffer(source: images[index], destination: readback, bytesPerRow: 256,
                                     destinationOffset: index < 4 ? 2 : 0),
                .read(buffer: readback, byteCount: index < 4 ? 4 : 8),
            ]
        }
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}
#endif
