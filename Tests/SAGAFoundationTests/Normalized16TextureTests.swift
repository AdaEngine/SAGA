import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private let normalized16Formats: [(TextureFormat, String)] = [
    (.r16Unorm, "r16unorm"), (.rg16Unorm, "rg16unorm"), (.rgba16Unorm, "rgba16unorm"),
    (.r16Snorm, "r16snorm"), (.rg16Snorm, "rg16snorm"), (.rgba16Snorm, "rgba16snorm"),
]

private func normalized16StorageShader(read: Bool, signed: Bool) -> String {
    let base = signed ? 3 : 0
    let declarations = normalized16Formats[base..<(base + 3)].enumerated().map { index, format in
        "@group(0) @binding(\(index)) var t\(index): texture_storage_2d<\(format.1), \(read ? "read" : "write")>;"
    }.joined(separator: "\n")
    let output = read ? "@group(0) @binding(3) var<storage, read_write> out: array<vec4f, 6>;" : ""
    let values = [
        "f32(x), 0, 0, 0", "0.25, 0.75, 0, 0", "-0.25, 0.25, 0.75, 2",
        "-0.5, 0, 0, 0", "-1, 1, 0, 0", "-2, -0.25, 0.25, 2",
    ]
    let statements = values[base..<(base + 3)].enumerated().map { index, value in
        read ? "out[\(base + index)] = textureLoad(t\(index), vec2u(1, 0));" : "textureStore(t\(index), vec2u(x, 0), vec4f(\(value)));"
    }.joined(separator: "\n")
    return """
        requires texture_formats_tier1;
        \(declarations)
        \(output)
        @compute @workgroup_size(\(read ? 1 : 2)) fn run(@builtin(local_invocation_index) x: u32) {
            \(statements)
        }
        """
}

private func normalized16SampleShader(linear: Bool) -> String {
    let declarations = normalized16Formats.indices.map { index in
        "@group(0) @binding(\(index)) var t\(index): texture_2d<f32>;"
    }.joined(separator: "\n")
    let x = linear ? "0.5" : "0.75"
    let statements = normalized16Formats.indices.map { index in
        "out[\(index)] = textureSampleLevel(t\(index), state, vec2f(\(x), 0.5), 0.0);"
    }.joined(separator: "\n")
    return """
        \(declarations)
        @group(0) @binding(6) var state: sampler;
        @group(0) @binding(7) var<storage, read_write> out: array<vec4f, 6>;
        @compute @workgroup_size(1) fn run() { \(statements) }
        """
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func normalized16StorageFormatsCompileWithCorrectTypes(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for signed in [false, true] {
        let base = signed ? 3 : 0
        for read in [false, true] {
            let source = ShaderInput.wgsl(normalized16StorageShader(read: read, signed: signed))
            #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(source) }
            _ = try compiler.parseAndValidate(source, capabilities: [.storageTexture16BitNormFormats])
            let shader = try compiler.compileCompute(source, to: format)
            #expect(shader.storageTextures.map { $0.format.rawValue.lowercased() } == normalized16Formats[base..<(base + 3)].map { $0.1 })
            #expect(shader.storageTextureFeatures == (read ? [.normalized16Textures, .storageTextureRead] : [.normalized16Textures]))
            guard case .text(_, let code) = shader.artifact else { Issue.record("Expected shader source"); return }
            if format == .wgsl {
                _ = try compiler.parseAndValidate(.wgsl(code), capabilities: [.storageTexture16BitNormFormats])
            }
            if format == .glsl { #expect(code.contains(signed ? "layout(r16_snorm," : "layout(rgba16,")) }
            if format == .hlsl { #expect(code.contains(signed ? "RWTexture2D<snorm float2>" : "RWTexture2D<unorm float4>")) }
        }
    }
}

@Test func normalized16TexturesRequireFeatureAndNearestUnlessFilterable() throws {
    let baseline = try DeviceCapabilities(maxBufferSize: 32)
    let available = try DeviceCapabilities(maxBufferSize: 32, features: [.normalized16Textures])
    for (format, _) in normalized16Formats {
        let texture = TextureDescriptor(width: 1, height: 1, format: format,
                                         usage: [.textureBinding, .renderAttachment, .storageBinding, .copySource])
        #expect(throws: GPUError.validation("Feature not advertised")) { try texture.validate(against: baseline) }
        try texture.validate(against: available)
        let image = ResourceID(device: 1, serial: 1)
        let slot = ComputeBindingSlot(binding: 0)
        let pipeline = ComputePipeline(id: .init(device: 1, serial: 2), storageBindings: [],
            textureBindings: [.init(slot: slot, sampleType: .float, requiresFiltering: true)])
        let dispatch = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1), textures: [slot: image])
        #expect(throws: GPUError.validation("Normalized 16-bit sampling requires normalized16Filterable")) {
            try dispatch.validate(device: 1, descriptors: [:], textures: [image: texture], features: available.features)
        }
        try dispatch.validate(device: 1, descriptors: [:], textures: [image: texture],
                              features: [.normalized16Textures, .normalized16Filterable])
    }
}

private let normalized16ColorShader = """
    struct Output { @location(0) red: f32, @location(1) rg: vec2f, @location(2) color: vec4f }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> Output {
        return Output(0.5, vec2f(-0.5, 0.75), vec4f(0.25, 0.5, 0.5, 0.5));
    }
    """

@Test func normalized16ResolveUsesTheDeviceCapability() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(normalized16ColorShader + """

        @fragment fn single() -> @location(0) f32 { return 0.5; }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "single", to: .msl)
    let capabilities = try DeviceCapabilities(maxBufferSize: 64, features: [.normalized16Textures])
    let pipeline = try RenderPipeline(id: .init(device: 1, serial: 1), colorFormat: .r16Unorm, vertexBuffers: [],
                                      sampleCount: 4, vertex: vertex, fragment: fragment, capabilities: capabilities)
    let target = ResourceID(device: 1, serial: 2)
    let resolved = ResourceID(device: 1, serial: 3)
    let textures: [ResourceID: TextureDescriptor] = [
        target: .init(width: 1, height: 1, format: .r16Unorm, usage: .renderAttachment, sampleCount: 4),
        resolved: .init(width: 1, height: 1, format: .r16Unorm, usage: .renderAttachment),
    ]
    let command = BufferCommand.render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                                       resolve: .init(texture: resolved))
    #expect(throws: GPUError.validation("Invalid color resolve attachment")) {
        try command.validate(device: 1, descriptors: [:], textures: textures, features: capabilities.features)
    }
    try command.validate(device: 1, descriptors: [:], textures: textures, features: [.normalized16Textures, .normalized16Resolve])
}

#if canImport(Metal)
@Test func writesReadsAndSamplesNormalized16TexturesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.normalized16Textures])
    let words = try await normalized16StorageDiagnostic(device)
    let expected: [Float] = [
        1, 0, 0, 1, 0.25, 0.75, 0, 1, 0, 0.25, 0.75, 1,
        -0.5, 0, 0, 1, -1, 1, 0, 1, -1, -0.25, 0.25, 1,
    ]
    let supportsLinear = device.capabilities.features.contains(.normalized16Filterable)
    let floats = words.dropFirst().map(Float.init(bitPattern:))
    let linear = [Float(0.5)] + Array(expected.dropFirst())
    #expect(words.first == 0xffff0000)
    let allExpected = expected + expected + (supportsLinear ? linear : [])
    #expect(floats.count == allExpected.count)
    for (actual, reference) in zip(floats, allExpected) { #expect(abs(actual - reference) < 0.0001) }
    await device.close()
}

private func normalized16StorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    var textures: [ResourceID] = []
    for (format, _) in normalized16Formats {
        textures.append(try await device.createTexture(.init(width: 2, height: 1, format: format,
                                                               usage: [.storageBinding, .textureBinding, .copySource])))
    }
    let images = Dictionary(uniqueKeysWithValues: textures.enumerated().map { (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element) })
    let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 96, usage: [.mapRead, .copyDestination]))
    let capture: [BufferCommand] = [.copy(source: output, destination: readback, byteCount: 96), .read(buffer: readback, byteCount: 96)]
    var commands: [BufferCommand] = []
    for signed in [false, true] {
        let writeShader = try compiler.compileCompute(.wgsl(normalized16StorageShader(read: false, signed: signed)), to: .msl)
        let readShader = try compiler.compileCompute(.wgsl(normalized16StorageShader(read: true, signed: signed)), to: .msl)
        let write = try await device.createComputePipeline(writeShader)
        let read = try await device.createComputePipeline(readShader)
        let base = signed ? 3 : 0
        let storage = Dictionary(uniqueKeysWithValues: textures[base..<(base + 3)].enumerated().map {
            (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element)
        })
        commands += [
            .dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: storage),
            .dispatchCompute(pipeline: read, bindings: [.init(binding: 3): output], workgroups: .init(x: 1), textures: storage),
        ]
    }
    commands += [
        .copyTextureToBuffer(source: textures[0], destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ] + capture
    for linear in [false, true] {
        let shader = try compiler.compileCompute(.wgsl(normalized16SampleShader(linear: linear)), to: .msl)
        let samplerSlot = ComputeBindingSlot(binding: 6)
        let pipeline = try await device.createComputePipeline(shader, nonFilteringSamplers: linear ? [] : [samplerSlot])
        let sampler = try await device.createSampler(.init(minFilter: linear ? .linear : .nearest, magFilter: linear ? .linear : .nearest))
        let dispatch = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 7): output],
                                                      workgroups: .init(x: 1), textures: images, samplers: [samplerSlot: sampler])
        if linear && !device.capabilities.features.contains(.normalized16Filterable) { continue }
        commands += [dispatch] + capture
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

@Test func blendsAndResolvesNormalized16ColorTargetsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.normalized16Textures])
    let words = try await normalized16ColorDiagnostic(device)
    let expected = [49151, -8192, 24575, 32768, 32768, 65535, 49151]
    #expect(words.count == 4)
    let lanes = [Int(words[0] >> 16), Int(Int16(bitPattern: UInt16(truncatingIfNeeded: words[1]))),
                 Int(Int16(bitPattern: UInt16(words[1] >> 16))), Int(words[2] & 0xffff), Int(words[2] >> 16),
                 Int(words[3] & 0xffff), Int(words[3] >> 16)]
    for (actual, reference) in zip(lanes, expected) { #expect(abs(actual - reference) <= 1) }
    await device.close()
}

private func normalized16ColorDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(normalized16ColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let resolve = device.capabilities.features.contains(.normalized16Resolve)
    let sampleCount: UInt32 = resolve ? 4 : 1
    let formats: [TextureFormat] = [.r16Unorm, .rg16Snorm, .rgba16Unorm]
    let additive = RenderBlendComponent(sourceFactor: .one, destinationFactor: .one)
    let blend = RenderColorState(blend: .init(color: additive, alpha: additive))
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: formats[0],
        sampleCount: sampleCount, colorState: blend, additionalColorTargets: [
            1: .init(format: formats[1], colorState: blend), 2: .init(format: formats[2], colorState: blend),
        ])
    var targets: [ResourceID] = []
    var outputs: [ResourceID] = []
    for format in formats {
        let target = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: resolve ? .renderAttachment : [.renderAttachment, .copySource], sampleCount: sampleCount))
        targets.append(target)
        outputs.append(resolve ? try await device.createTexture(.init(width: 1, height: 1, format: format,
                                      usage: [.renderAttachment, .copySource])) : target)
    }
    let readback = try await device.createBuffer(.init(size: 8, usage: [.mapRead, .copyDestination]))
    let clear = RenderClearColor(red: 0.25, green: 0, blue: 0.5, alpha: 0.25)
    return try await device.submitCommands([
        .render(pipeline: pipeline, target: targets[0], clear: clear, vertexCount: 3,
                resolve: resolve ? .init(texture: outputs[0]) : nil, additionalColors: [
                    1: .init(texture: targets[1], loadAction: .clear(clear), resolve: resolve ? .init(texture: outputs[1]) : nil),
                    2: .init(texture: targets[2], loadAction: .clear(clear), resolve: resolve ? .init(texture: outputs[2]) : nil),
                ]),
        .copyTextureToBuffer(source: outputs[0], destination: readback, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: outputs[1], destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: outputs[2], destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
    ]).wait().flatMap { $0 }
}
#endif
