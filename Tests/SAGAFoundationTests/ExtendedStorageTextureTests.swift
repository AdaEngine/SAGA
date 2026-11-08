import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private let bgraStorageShader = """
    @group(0) @binding(0) var image: texture_storage_2d<bgra8unorm, write>;
    @compute @workgroup_size(1) fn run() {
        textureStore(image, vec2u(0), vec4f(0.25, 0.5, 0.75, 1.0));
    }
    """

private let bgraSampleShader = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() { out = textureLoad(image, vec2u(0), 0u); }
    """

private let wideStorageShader = """
    @group(0) @binding(0) var halfImage: texture_storage_2d<rgba16float, read_write>;
    @group(0) @binding(1) var fullImage: texture_storage_2d<rgba32float, read_write>;
    @group(0) @binding(2) var uintImage: texture_storage_2d<rgba32uint, read_write>;
    @group(0) @binding(3) var sintImage: texture_storage_2d<rgba32sint, read_write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(halfImage, p, textureLoad(halfImage, p) + vec4f(0.5, -1, 2, -2));
        textureStore(fullImage, p, textureLoad(fullImage, p) - vec4f(0.5, -0.25, 0.75, -1.25));
        textureStore(uintImage, p, textureLoad(uintImage, p) + vec4u(1, 2, 3, 4));
        textureStore(sintImage, p, textureLoad(sintImage, p) + vec4i(-1, 2, -3, 4));
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func wideRGBAStorageUsesReadWriteLayoutAndDeviceFeature(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(wideStorageShader), to: format)
    #expect(shader.storageTextures.map(\.format) == [.rgba16Float, .rgba32Float, .rgba32Uint, .rgba32Sint])
    #expect(shader.storageTextures.allSatisfy { $0.access == [.load, .store] })
    #expect(shader.storageTextureFeatures == [.storageTextureReadWrite, .wideRGBAStorageReadWrite])
    if format == .wgsl, case .text(_, let code) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(code))
    }
}

@Test func bgraStorageIsWriteOnlyAndHasNoNativeGLSLImageLayout() throws {
    let compiler = IndependentShaderCompiler()
    let descriptor = TextureDescriptor(width: 1, height: 1, format: .bgra8Unorm,
                                       usage: [.storageBinding, .textureBinding, .copySource])
    #expect(throws: GPUError.validation("Feature not advertised")) {
        try descriptor.validate(against: .init(maxBufferSize: 64))
    }
    try descriptor.validate(against: .init(maxBufferSize: 64, features: [.bgra8UnormStorage]))
    for format: ShaderOutputFormat in [.msl, .hlsl, .wgsl] {
        let shader = try compiler.compileCompute(.wgsl(bgraStorageShader), to: format)
        #expect(shader.storageTextures.map(\.format) == [.bgra8Unorm])
        #expect(shader.storageTextures.map(\.access) == [.store])
        #expect(shader.storageTextureFeatures == [.bgra8UnormStorage])
    }
    do {
        _ = try compiler.compileCompute(.wgsl(bgraStorageShader), to: .glsl)
        Issue.record("Expected BGRA8 storage to fail for GLSL")
    } catch ShaderFailure.unsupported(let message) {
        #expect(message.contains("BGRA8"))
    }
    for access in ["read", "read_write"] {
        #expect(throws: ShaderFailure.self) {
            try compiler.parseAndValidate(.wgsl("@group(0) @binding(0) var image: texture_storage_2d<bgra8unorm, \(access)>;"))
        }
    }
}

#if canImport(Metal)
@Test func bgraStorageWritesChannelsInTheirPhysicalOrderOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.bgra8UnormStorage])
    let words = try await bgraStorageDiagnostic(device)
    #expect(words.first == 0xff4080bf)
    let expected: [Float] = [64.0 / 255, 128.0 / 255, 191.0 / 255, 1]
    let values = words.dropFirst().map(Float.init(bitPattern:))
    #expect(values.count == expected.count)
    for (actual, reference) in zip(values, expected) { #expect(abs(actual - reference) < 0.0001) }
    await device.close()
}

private func bgraStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let write = try await device.createComputePipeline(compiler.compileCompute(.wgsl(bgraStorageShader), to: .msl))
    let sample = try await device.createComputePipeline(compiler.compileCompute(.wgsl(bgraSampleShader), to: .msl))
    let image = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8Unorm,
                                                     usage: [.storageBinding, .textureBinding, .copySource]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let bindings: [ComputeBindingSlot: ResourceID] = [.init(binding: 0): image]
    return try await device.submitCommands([
        .dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: bindings),
        .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .dispatchCompute(pipeline: sample, bindings: [.init(binding: 1): output], workgroups: .init(x: 1), textures: bindings),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]).wait().flatMap { $0 }
}

@Test func updatesAllWideRGBAStorageFormatsInPlaceOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    if device.capabilities.features.contains(.wideRGBAStorageReadWrite) {
        let words = try await wideStorageDiagnostic(device)
        let expected: [UInt32] = [0x3c003e00, 0x40004500] +
            [Float(-1.5), 2.25, -3.75, 5.25].map(\.bitPattern) + [101, 202, 303, 404] +
            [Int32(-11), 22, -33, 44].map(UInt32.init(bitPattern:))
        #expect(words == expected)
    }
    await device.close()
}

private func wideStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(wideStorageShader), to: .msl)
    let compute = try await device.createComputePipeline(shader)
    let formats: [TextureFormat] = [.rgba16Float, .rgba32Float, .rgba32Uint, .rgba32Sint]
    let data = [
        [UInt32(0x40003c00), 0x44004200],
        [Float(-1), 2, -3, 4].map(\.bitPattern),
        [100, 200, 300, 400],
        [Int32(-10), 20, -30, 40].map(UInt32.init(bitPattern:)),
    ]
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    var textures: [ResourceID] = []
    var commands: [BufferCommand] = []
    for (format, words) in zip(formats, data) {
        let image = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: [.storageBinding, .copySource, .copyDestination]))
        textures.append(image)
        commands.append(.writeTexture(destination: image, data: words.withUnsafeBytes { Array($0) },
                                      region: .init(width: 1, height: 1)))
    }
    let bindings = Dictionary(uniqueKeysWithValues: textures.enumerated().map {
        (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element)
    })
    commands.append(.dispatchCompute(pipeline: compute, bindings: [:], workgroups: .init(x: 1), textures: bindings))
    for (index, image) in textures.enumerated() {
        commands += [.copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
                     .read(buffer: readback, byteCount: index == 0 ? 8 : 16)]
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}
#endif
