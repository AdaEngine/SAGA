import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private struct ByteTextureCase: Sendable {
    let format: TextureFormat
    let name: String
    let scalar: String
    let vector: String
    let bytes: [UInt8]
    let value: String
}

private let byteTextureCases: [ByteTextureCase] = [
    .init(format: .r8Unorm, name: "r8unorm", scalar: "f32", vector: "vec4f", bytes: [128], value: "128.0/255.0, 0, 0, 0"),
    .init(format: .rg8Unorm, name: "rg8unorm", scalar: "f32", vector: "vec4f", bytes: [64, 192], value: "64.0/255.0, 192.0/255.0, 0, 0"),
    .init(format: .r8Snorm, name: "r8snorm", scalar: "f32", vector: "vec4f", bytes: [192], value: "-64.0/127.0, 0, 0, 0"),
    .init(format: .rg8Snorm, name: "rg8snorm", scalar: "f32", vector: "vec4f", bytes: [193, 127], value: "-63.0/127.0, 1, 0, 0"),
    .init(format: .r8Uint, name: "r8uint", scalar: "u32", vector: "vec4u", bytes: [250], value: "250, 0, 0, 0"),
    .init(format: .rg8Uint, name: "rg8uint", scalar: "u32", vector: "vec4u", bytes: [4, 200], value: "4, 200, 0, 0"),
    .init(format: .r8Sint, name: "r8sint", scalar: "i32", vector: "vec4i", bytes: [129], value: "-127, 0, 0, 0"),
    .init(format: .rg8Sint, name: "rg8sint", scalar: "i32", vector: "vec4i", bytes: [128, 100], value: "-128, 100, 0, 0"),
]

private func byteReadShader(storage: Bool, base: Int, count: Int) -> String {
    let declarations = byteTextureCases[base..<(base + count)].enumerated().map { index, entry in
        let type = storage ? "texture_storage_2d<\(entry.name), read>" : "texture_2d<\(entry.scalar)>"
        return "@group(0) @binding(\(index)) var t\(index): \(type);"
    }.joined(separator: "\n")
    let expressions = (0..<count).map { index in
        "out[\(base + index)] = vec4f(textureLoad(t\(index), vec2u(0)\(storage ? "" : ", 0u")));"
    }.joined(separator: "\n")
    return """
        \(storage ? "requires texture_formats_tier1;" : "")
        \(declarations)
        @group(0) @binding(\(count)) var<storage, read_write> out: array<vec4f, 8>;
        @compute @workgroup_size(1) fn run() { \(expressions) }
        """
}

private func byteWriteShader(base: Int) -> String {
    let cases = byteTextureCases[base..<(base + 4)]
    let declarations = cases.enumerated().map { index, entry in
        "@group(0) @binding(\(index)) var t\(index): texture_storage_2d<\(entry.name), write>;"
    }.joined(separator: "\n")
    let statements = cases.enumerated().map { index, entry in
        "textureStore(t\(index), vec2u(0), \(entry.vector)(\(entry.value)));"
    }.joined(separator: "\n")
    return """
        requires texture_formats_tier1;
        \(declarations)
        @compute @workgroup_size(1) fn run() { \(statements) }
        """
}

private let byteUpdateShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var narrow: texture_storage_2d<r8uint, read_write>;
    @group(0) @binding(1) var rgba: texture_storage_2d<rgba8unorm, read_write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(narrow, p, textureLoad(narrow, p) + vec4u(3, 0, 0, 0));
        textureStore(rgba, p, textureLoad(rgba, p) + vec4f(1, 0, 0, 1));
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func narrow8BitStorageEmitsCorrectAccessAndFeatureGates(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for base in [0, 4] {
        let write = try compiler.compileCompute(.wgsl(byteWriteShader(base: base)), to: format)
        let read = try compiler.compileCompute(.wgsl(byteReadShader(storage: true, base: base, count: 4)), to: format)
        #expect(write.storageTextures.map { $0.format.rawValue.lowercased() } == byteTextureCases[base..<(base + 4)].map(\.name))
        #expect(write.storageTextureFeatures == [.narrow8BitStorageTextures])
        #expect(read.storageTextureFeatures == [.narrow8BitStorageTextures, .storageTextureRead])
    }
    let update = try compiler.compileCompute(.wgsl(byteUpdateShader), to: format)
    #expect(update.storageTextureFeatures == [.narrow8BitStorageTextures, .storage8BitReadWrite, .storageTextureReadWrite])
}

@Test func byteTexturesHaveCoreSamplingAndFeatureGatedRenderAndStorage() throws {
    let core = try DeviceCapabilities(maxBufferSize: 64)
    let tier1 = try DeviceCapabilities(maxBufferSize: 64, features: [.snorm8Renderable, .narrow8BitStorageTextures])
    for entry in byteTextureCases {
        let format = entry.format
        try TextureDescriptor(width: 1, height: 1, format: format, usage: [.textureBinding, .copySource, .copyDestination])
            .validate(against: core)
        let storage = TextureDescriptor(width: 1, height: 1, format: format, usage: .storageBinding)
        #expect(throws: GPUError.validation("Feature not advertised")) { try storage.validate(against: core) }
        try storage.validate(against: tier1)
        let color = TextureDescriptor(width: 1, height: 1, format: format, usage: .renderAttachment, sampleCount: 4)
        if entry.name.hasSuffix("snorm") {
            #expect(throws: GPUError.validation("Feature not advertised")) { try color.validate(against: core) }
        } else {
            try color.validate(against: core)
        }
        try color.validate(against: tier1)
    }
}

private let byteColorShader = """
    struct Targets { @location(0) r: f32, @location(1) rg: vec2f, @location(2) rgba: vec4f }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn unorm() -> Targets {
        return Targets(128.0/255.0, vec2f(64.0/255.0, 192.0/255.0), vec4f(1, 0, 0, 1));
    }
    @fragment fn snorm() -> Targets {
        return Targets(-64.0/127.0, vec2f(-63.0/127.0, 1), vec4f(-64.0/127.0, 1, 0, 1));
    }
    """

#if canImport(Metal)
@Test func transfersSamplesAndResolvesByteFormatsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let transfer = try await byteTransferDiagnostic(device)
    #expect(transfer.first == 0xaabb80dd)
    let expected: [Float] = [
        128.0/255.0, 0, 0, 1, 64.0/255.0, 192.0/255.0, 0, 1,
        -64.0/127.0, 0, 0, 1, -63.0/127.0, 1, 0, 1,
        250, 0, 0, 1, 4, 200, 0, 1, -127, 0, 0, 1, -128, 100, 0, 1,
    ]
    let samples = transfer.dropFirst().map(Float.init(bitPattern:))
    #expect(samples.count == expected.count)
    for (actual, reference) in zip(samples, expected) { #expect(abs(actual - reference) < 0.0001) }
    let colors = try await byteColorDiagnostic(device)
    let snorm: [UInt32] = device.capabilities.features.contains(.snorm8Renderable) ? [0xc0000000, 0x7fc10000, 0x7f007fc0] : []
    #expect(colors == [0x80000000, 0xc0400000, 0xff0000ff] + snorm)
    await device.close()
}

private func byteTransferDiagnostic(_ device: Device) async throws -> [UInt32] {
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let upload = try await device.createBuffer(.init(size: 4, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 128, usage: [.mapRead, .copyDestination]))
    let shader = try IndependentShaderCompiler().compileCompute(
        .wgsl(byteReadShader(storage: false, base: 0, count: 8)), to: .msl)
    let compute = try await device.createComputePipeline(shader)
    var images: [ComputeBindingSlot: ResourceID] = [:]
    var commands: [BufferCommand] = [.write(buffer: upload, offset: 0, words: [0x00008000])]
    for (index, entry) in byteTextureCases.enumerated() {
        let image = try await device.createTexture(.init(width: 1, height: 1, format: entry.format, usage: usage))
        images[.init(binding: UInt32(index))] = image
        let region = TextureCopyRegion(width: 1, height: 1)
        if index == 0 {
            commands += [
                .copyBufferToTexture(source: upload, layout: .init(offset: 1), destination: image, region: region),
                .write(buffer: readback, offset: 0, words: [0xaabbccdd]),
                .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256, destinationOffset: 1),
                .read(buffer: readback, byteCount: 4),
            ]
        } else {
            commands.append(.writeTexture(destination: image, data: [0xee] + entry.bytes, layout: .init(offset: 1), region: region))
        }
    }
    commands += [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 8): output], workgroups: .init(x: 1), textures: images),
        .copy(source: output, destination: readback, byteCount: 128), .read(buffer: readback, byteCount: 128),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private func byteColorDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(byteColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    var commands: [BufferCommand] = []
    for snorm in [false, true] {
        if snorm && !device.capabilities.features.contains(.snorm8Renderable) { continue }
        let formats: [TextureFormat] = snorm ? [.r8Snorm, .rg8Snorm, .rgba8Snorm] : [.r8Unorm, .rg8Unorm, .rgba8Unorm]
        let fragment = try compiler.compileRenderStage(source, entryPoint: snorm ? "snorm" : "unorm", to: .msl)
        let blend = RenderColorState(blend: .init())
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: formats[0],
            sampleCount: 4, colorState: blend, additionalColorTargets: [
                1: .init(format: formats[1], colorState: blend), 2: .init(format: formats[2], colorState: blend),
            ])
        var targets: [ResourceID] = []
        var resolves: [ResourceID] = []
        for format in formats {
            targets.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                                usage: .renderAttachment, sampleCount: 4)))
            resolves.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                                 usage: [.renderAttachment, .copySource])))
        }
        commands.append(.render(pipeline: pipeline, target: targets[0], clear: .transparentBlack, vertexCount: 3,
            resolve: .init(texture: resolves[0]), additionalColors: [
                1: .init(texture: targets[1], loadAction: .clear(.transparentBlack), resolve: .init(texture: resolves[1])),
                2: .init(texture: targets[2], loadAction: .clear(.transparentBlack), resolve: .init(texture: resolves[2])),
            ]))
        for (image, offset) in zip(resolves, [UInt64(3), 2, 0]) {
            commands += [
                .write(buffer: readback, offset: 0, words: [0]),
                .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256, destinationOffset: offset),
                .read(buffer: readback, byteCount: 4),
            ]
        }
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

@Test func writesReadsAndUpdatesByteStorageFormatsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.narrow8BitStorageTextures])
    let words = try await byteStorageDiagnostic(device)
    let expected: [Float] = [
        128.0/255.0, 0, 0, 1, 64.0/255.0, 192.0/255.0, 0, 1,
        -64.0/127.0, 0, 0, 1, -63.0/127.0, 1, 0, 1,
        250, 0, 0, 1, 4, 200, 0, 1, -127, 0, 0, 1, -128, 100, 0, 1,
    ]
    let samples = words.prefix(32).map(Float.init(bitPattern:))
    #expect(samples.count == expected.count)
    for (actual, reference) in zip(samples, expected) { #expect(abs(actual - reference) < 0.0001) }
    let updated: [UInt32] = device.capabilities.features.contains(.storage8BitReadWrite) ? [0xfd000000, 0xff0000ff] : []
    #expect(Array(words.dropFirst(32)) == updated)
    await device.close()
}

private func byteStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    var images: [ResourceID] = []
    for entry in byteTextureCases {
        images.append(try await device.createTexture(.init(width: 1, height: 1, format: entry.format,
                                                             usage: [.storageBinding, .copySource])))
    }
    let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 128, usage: [.mapRead, .copyDestination]))
    var commands: [BufferCommand] = []
    for base in [0, 4] {
        let write = try await device.createComputePipeline(compiler.compileCompute(.wgsl(byteWriteShader(base: base)), to: .msl))
        let read = try await device.createComputePipeline(compiler.compileCompute(
            .wgsl(byteReadShader(storage: true, base: base, count: 4)), to: .msl))
        let bindings = Dictionary(uniqueKeysWithValues: images[base..<(base + 4)].enumerated().map {
            (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element)
        })
        commands += [
            .dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: bindings),
            .dispatchCompute(pipeline: read, bindings: [.init(binding: 4): output], workgroups: .init(x: 1), textures: bindings),
        ]
    }
    commands += [.copy(source: output, destination: readback, byteCount: 128), .read(buffer: readback, byteCount: 128)]
    if device.capabilities.features.contains(.storage8BitReadWrite) {
        let rgba = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.storageBinding, .copySource]))
        let update = try await device.createComputePipeline(compiler.compileCompute(.wgsl(byteUpdateShader), to: .msl))
        commands += [
            .dispatchCompute(pipeline: update, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): images[4], .init(binding: 1): rgba]),
            .write(buffer: readback, offset: 0, words: [0]),
            .copyTextureToBuffer(source: images[4], destination: readback, bytesPerRow: 256, destinationOffset: 3),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: rgba, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}
#endif
