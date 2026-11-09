import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

@Test func halfFloatFormatsUseTheirExactTexelFootprints() throws {
    let capabilities = try DeviceCapabilities(maxBufferSize: 64)
    let texture = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    for (format, bytes) in [(TextureFormat.r16Float, UInt64(2)), (.rg16Float, 4), (.rgba16Float, 8)] {
        let descriptor = TextureDescriptor(width: 1, height: 1, format: format,
                                           usage: [.copySource, .copyDestination, .textureBinding, .renderAttachment])
        try descriptor.validate(against: capabilities)
        try TextureDescriptor(width: 1, height: 1, format: format, usage: .renderAttachment, sampleCount: 4)
            .validate(against: capabilities)
        let buffers = [buffer: BufferDescriptor(size: bytes * 2, usage: [.copySource, .copyDestination])]
        let textures = [texture: descriptor]
        try BufferCommand.copyBufferToTexture(source: buffer, layout: .init(offset: bytes), destination: texture,
                                               region: .init(width: 1, height: 1))
            .validate(device: 1, descriptors: buffers, textures: textures)
        try BufferCommand.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256,
                                               destinationOffset: bytes)
            .validate(device: 1, descriptors: buffers, textures: textures)
        #expect(throws: GPUError.self) {
            try BufferCommand.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256,
                                                  destinationOffset: bytes - 1)
                .validate(device: 1, descriptors: buffers, textures: textures)
        }
    }
    for format: TextureFormat in [.r16Float, .rg16Float] {
        #expect(throws: GPUError.validation("Feature not advertised")) {
            try TextureDescriptor(width: 1, height: 1, format: format, usage: .storageBinding).validate(against: capabilities)
        }
    }
}

private let halfColorShader = """
    struct Outputs {
        @location(0) red: f32,
        @location(1) pair: vec2f,
        @location(2) color: vec4f,
    }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> Outputs {
        return Outputs(2.5, vec2f(-1, 3), vec4f(0.5, -0.5, 4, 0.25));
    }
    """

private let halfSampleShader = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var state: sampler;
    @group(0) @binding(2) var<storage, read_write> out: f32;
    @compute @workgroup_size(1) fn run() { out = textureSampleLevel(image, state, vec2f(0.5), 0.0).x; }
    """

@Test func mixedHalfAndFullFloatTargetsAlignTheColorBudgetByLocation() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(halfColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let targets: [UInt32: RenderColorTarget] = [
        1: .init(format: .rg32Float),
        2: .init(format: .rgba32Float),
        3: .init(format: .rg16Float, colorState: .init(writeMask: [])),
        4: .init(format: .r16Float, colorState: .init(writeMask: [])),
    ]
    // The unaligned total is 32. Aligning location one adds two bytes.
    #expect(throws: GPUError.validation("Color attachments exceed the device bytes-per-sample limit")) {
        try RenderPipeline(id: .init(device: 1, serial: 1), colorFormat: .r16Float, vertexBuffers: [],
                           additionalColorTargets: targets, vertex: vertex, fragment: fragment)
    }
}

#if canImport(Metal)
@Test func uploadsAndCopiesHalfTexelsAtTwoByteOffsetsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let snapshots = try await halfTexelTransferDiagnostic(device)
    let expected: [[UInt32]] = [
        [0x00001234], [0x42001234], [0x3c001234], [Float(2).bitPattern],
    ]
    #expect(snapshots == expected)
    await device.close()
}

private func halfTexelTransferDiagnostic(_ device: Device) async throws -> [[UInt32]] {
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let line = try await device.createTexture(.init(width: 1, height: 1, format: .r16Float, usage: usage, dimension: .d1))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .r16Float, usage: usage))
    let upload = try await device.createBuffer(.init(size: 4, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    let linear = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfSampleShader), to: .msl)
    let pipeline = try await device.createComputePipeline(shader)
    return try await device.submitCommands([
        .write(buffer: read, offset: 0, words: [0xdead1234]),
        .copyTextureToBuffer(source: line, destination: read, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: read, byteCount: 4),
        .write(buffer: upload, offset: 0, words: [0x4200ffff]),
        .copyBufferToTexture(source: upload, layout: .init(offset: 2), destination: line, region: .init(width: 1, height: 1)),
        .copyTextureToBuffer(source: line, destination: read, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: read, byteCount: 4),
        .writeTexture(destination: line, data: [0x99, 0x00, 0x3c], layout: .init(offset: 1),
                      region: .init(width: 1, height: 1)),
        .copyTextureToBuffer(source: line, destination: read, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: read, byteCount: 4),
        .writeTexture(destination: image, data: [0x00, 0x3c, 0x00, 0x42], region: .init(width: 2, height: 1)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): image], samplers: [.init(binding: 1): linear]),
        .copy(source: output, destination: read, byteCount: 4), .read(buffer: read, byteCount: 4),
    ]).wait()
}

@Test func blendsAndResolvesThreeHalfFloatFormatsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let words = try await halfColorResolveDiagnostic(device)
    let expected: [[UInt32]] = [[0x43000000], [0x3c000000], [0xc1003e00, 0x3a004700]]
    #expect(words == expected)
    await device.close()
}

private func halfColorResolveDiagnostic(_ device: Device) async throws -> [[UInt32]] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(halfColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let formats: [TextureFormat] = [.r16Float, .rg16Float, .rgba16Float]
    let additive = RenderBlendComponent(sourceFactor: .one, destinationFactor: .one)
    let state = RenderColorState(blend: .init(color: additive, alpha: additive))
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: formats[0], sampleCount: 4, colorState: state,
        additionalColorTargets: [1: .init(format: formats[1], colorState: state),
                                 2: .init(format: formats[2], colorState: state)])
    var targets: [ResourceID] = []
    var resolves: [ResourceID] = []
    for format in formats {
        targets.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                             usage: .renderAttachment, sampleCount: 4)))
        resolves.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                              usage: [.renderAttachment, .copySource])))
    }
    let read = try await device.createBuffer(.init(size: 8, usage: [.mapRead, .copyDestination]))
    let clear = RenderClearColor(red: 1, green: -2, blue: 3, alpha: 0.5)
    var commands: [BufferCommand] = [
        .render(pipeline: pipeline, target: targets[0], clear: clear, vertexCount: 3, resolve: .init(texture: resolves[0]),
                additionalColors: [
                    1: .init(texture: targets[1], loadAction: .clear(clear), resolve: .init(texture: resolves[1])),
                    2: .init(texture: targets[2], loadAction: .clear(clear), resolve: .init(texture: resolves[2])),
                ]),
        .copyTextureToBuffer(source: resolves[0], destination: read, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: resolves[1], destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 4),
    ]
    commands += [.copyTextureToBuffer(source: resolves[2], destination: read, bytesPerRow: 256),
                 .read(buffer: read, byteCount: 8)]
    return try await device.submitCommands(commands).wait()
}
#endif
