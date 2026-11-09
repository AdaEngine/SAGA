import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private let float32ColorShader = """
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
    @fragment fn one() -> @location(0) f32 { return 2.5; }
    @fragment fn two() -> @location(0) vec2f { return vec2f(-1, 3); }
    @fragment fn four() -> @location(0) vec4f { return vec4f(0.5, -0.5, 4, 0.25); }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsNarrowFragmentColorOutputs(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fragment = try compiler.compileRenderStage(.wgsl(float32ColorShader), entryPoint: "fs", to: format)
    #expect(fragment.colorOutputComponentCounts == [0: 1, 1: 2, 2: 4])
    #expect(fragment.colorOutputKinds == [0: .float, 1: .float, 2: .float])
    let scalar = try compiler.compileRenderStage(.wgsl(float32ColorShader), entryPoint: "one", to: format)
    #expect(scalar.colorOutputComponentCounts == [0: 1])
    if format == .wgsl {
        guard case .text(_, let source) = scalar.artifact else { Issue.record("Expected WGSL"); return }
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test func float32ColorTargetValidationChecksComponentCountFeatureAndBudget() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(float32ColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let id = ResourceID(device: 1, serial: 1)
    let targets: [UInt32: RenderColorTarget] = [1: .init(format: .rg32Float), 2: .init(format: .rgba32Float)]
    _ = try RenderPipeline(id: id, colorFormat: .r32Float, vertexBuffers: [],
                           additionalColorTargets: targets, vertex: vertex, fragment: fragment)
    #expect(throws: GPUError.validation("Fragment output has too few components for its color attachment")) {
        try RenderPipeline(id: id, colorFormat: .rg32Float, vertexBuffers: [],
                           additionalColorTargets: targets, vertex: vertex, fragment: fragment)
    }
    #expect(throws: GPUError.validation("Color attachments exceed the device bytes-per-sample limit")) {
        try RenderPipeline(id: id, colorFormat: .r32Float, vertexBuffers: [],
                           additionalColorTargets: [1: .init(format: .rgba32Float), 2: .init(format: .rgba32Float)],
                           vertex: vertex, fragment: fragment)
    }
    let blendable = try DeviceCapabilities(maxBufferSize: 256, features: [.float32Blendable])
    let additive = RenderBlendComponent(sourceFactor: .one, destinationFactor: .one)
    let blend = RenderColorState(blend: .init(color: additive, alpha: additive))
    #expect(throws: GPUError.validation("Float32 blending requires float32-blendable")) {
        try RenderPipeline(id: id, colorFormat: .r32Float, vertexBuffers: [], colorState: blend,
                           additionalColorTargets: targets, vertex: vertex, fragment: fragment)
    }
    _ = try RenderPipeline(id: id, colorFormat: .r32Float, vertexBuffers: [], colorState: blend,
                           additionalColorTargets: targets, vertex: vertex, fragment: fragment, capabilities: blendable)
    let sourceAlpha = RenderColorState(blend: .init(color: .init(sourceFactor: .sourceAlpha)))
    #expect(throws: GPUError.validation("Source alpha blending requires a four-component fragment output")) {
        try RenderPipeline(id: id, colorFormat: .r32Float, vertexBuffers: [], colorState: sourceAlpha,
                           additionalColorTargets: targets, vertex: vertex, fragment: fragment, capabilities: blendable)
    }

    let capabilities = try DeviceCapabilities(maxBufferSize: 256)
    for format: TextureFormat in [.r32Float, .rg32Float, .rgba32Float] {
        try TextureDescriptor(width: 1, height: 1, format: format, usage: [.renderAttachment, .copySource])
            .validate(against: capabilities)
    }
    let msaa = try DeviceCapabilities(maxBufferSize: 256, multisampledFloat32Formats: [.r32Float, .rg32Float])
    for format: TextureFormat in [.r32Float, .rg32Float, .rgba32Float] {
        let descriptor = TextureDescriptor(width: 1, height: 1, format: format,
                                           usage: [.renderAttachment, .textureBinding], sampleCount: 4)
        #expect(throws: GPUError.self) { try descriptor.validate(against: capabilities) }
        if format != .rgba32Float {
            try descriptor.validate(against: msaa)
        } else {
            #expect(throws: GPUError.self) { try descriptor.validate(against: msaa) }
        }
    }
    let scalar = try compiler.compileRenderStage(source, entryPoint: "one", to: .msl)
    #expect(throws: GPUError.validation("Color attachment format does not support multisampling")) {
        try RenderPipeline(id: id, colorFormat: .r32Float, vertexBuffers: [], sampleCount: 4,
                           vertex: vertex, fragment: scalar, capabilities: capabilities)
    }
    _ = try RenderPipeline(id: id, colorFormat: .r32Float, vertexBuffers: [], sampleCount: 4,
                           vertex: vertex, fragment: scalar, capabilities: msaa)
}

#if canImport(Metal)
@Test func rendersAndBlendsThreeFloat32ColorFormatsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.float32Blendable])
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(float32ColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let formats: [TextureFormat] = [.r32Float, .rg32Float, .rgba32Float]
    var images: [ResourceID] = []
    for format in formats {
        images.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                            usage: [.renderAttachment, .copySource])))
    }
    let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 1, green: -2, blue: 3, alpha: 0.5)
    let additional: [UInt32: RenderColorAttachment] = [
        1: .init(texture: images[1], loadAction: .clear(clear)),
        2: .init(texture: images[2], loadAction: .clear(clear)),
    ]
    let additive = RenderBlendComponent(sourceFactor: .one, destinationFactor: .one)
    var snapshots: [[UInt32]] = []
    for blend in [false, true] {
        let state = RenderColorState(blend: blend ? .init(color: additive, alpha: additive) : nil)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: formats[0], colorState: state,
            additionalColorTargets: [1: .init(format: formats[1], colorState: state),
                                     2: .init(format: formats[2], colorState: state)])
        var commands: [BufferCommand] = [
            .render(pipeline: pipeline, target: images[0], clear: clear, vertexCount: 3, additionalColors: additional),
        ]
        for (image, bytes) in zip(images, [UInt64(4), 8, 16]) {
            commands += [.copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
                         .read(buffer: readback, byteCount: bytes)]
        }
        snapshots += try await device.submitCommands(commands).wait()
    }
    let expected: [[Float]] = [[2.5], [-1, 3], [0.5, -0.5, 4, 0.25], [3.5], [0, 1], [1.5, -2.5, 7, 0.75]]
    #expect(snapshots == expected.map { $0.map(\.bitPattern) })
    await device.close()
}

private let float32MultisampleShader = """
    @group(0) @binding(0) var image: texture_multisampled_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let texel = textureLoad(image, vec2u(0), 3u);
        out = vec4f(texel.xyz, f32(textureNumSamples(image)));
    }
    """

@Test(arguments: [
    (TextureFormat.r32Float, "one", [Float(2.5), 0, 0, 4]),
    (.rg32Float, "two", [-1, 3, 0, 4]),
    (.rgba32Float, "four", [0.5, -0.5, 4, 4]),
])
func rendersAndReadsSupportedMultisampledFloat32OnMetal(format: TextureFormat, entry: String, expected: [Float]) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    if device.capabilities.multisampledFloat32Formats.contains(format) {
        let snapshots = try await renderMultisampledFloat32(device, format: format, entry: entry)
        #expect(snapshots == expected.map(\.bitPattern))
    }
    await device.close()
}

private func renderMultisampledFloat32(_ device: Device, format: TextureFormat, entry: String) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(float32ColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: entry, to: .msl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: format, sampleCount: 4)
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(float32MultisampleShader), to: .msl))
    let image = try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                     usage: [.renderAttachment, .textureBinding], sampleCount: 4))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    return try await device.submitCommands([
        .render(pipeline: render, target: image, clear: .transparentBlack, vertexCount: 3),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 1): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): image]),
        .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ]).wait().flatMap { $0 }
}
#endif
