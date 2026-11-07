import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let multisampleReader = """
    @group(0) @binding(0) var color: texture_multisampled_2d<f32>;
    @group(0) @binding(1) var depth: texture_depth_multisampled_2d;
    @group(0) @binding(2) var<storage, read_write> out: array<u32, 12>;
    @compute @workgroup_size(1) fn run() {
        for (var lane = 0u; lane < 4u; lane++) {
            let r = textureLoad(color, vec2u(0), lane).r;
            let d = textureLoad(depth, vec2i(0), i32(lane));
            out[lane] = select(select(2u, 0u, r == 0.0), 1u, r == 1.0);
            out[4u + lane] = select(select(0u, 4u, d == 1.0), 1u, d == 0.25);
        }
        let dimensions = textureDimensions(color);
        let negative = textureLoad(color, vec2i(0), -1i);
        let outside = textureLoad(depth, vec2u(0xffffffffu), 0u);
        let invalid = textureLoad(depth, vec2u(0), 0xffffffffu);
        out[8] = textureNumSamples(color);
        out[9] = textureNumSamples(depth);
        out[10] = dimensions.x + dimensions.y;
        out[11] = select(1u, 0u, negative.a == 0.0 && outside == 0.0 && invalid == 0.0);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsMultisampledTextureLoadsAndQueries(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(multisampleReader), to: format)
    #expect(shader.sampledTextures.map(\.isDepth) == [false, true])
    #expect(shader.sampledTextures.map(\.isMultisampled) == [true, true])
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("texture_multisampled_2d<f32>"))
        #expect(source.contains("texture_depth_multisampled_2d"))
        #expect(source.contains("textureNumSamples("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains("texture2d_ms<float, access::read>"))
        #expect(source.contains("depth2d_ms<float, access::read>"))
        #expect(source.contains(".get_num_samples()"))
    case .hlsl:
        #expect(source.contains("Texture2DMS<float4>") && source.contains("Texture2DMS<float>"))
        #expect(source.contains(".Load(int2(") && source.contains(".GetDimensions("))
    case .glsl:
        #expect(source.contains("sampler2DMS") && source.contains("textureSamples("))
        #expect(source.contains("texelFetch(") && source.contains(").r;"))
    default: Issue.record("Unexpected shader format")
    }
}

@Test(arguments: [
    "textureLoad(color, vec2u(0), 0.0)",
    "textureLoad(depth, vec2u(0))",
    "textureDimensions(color, 0)",
    "textureNumLevels(depth)",
    "textureNumSamples(single)",
    "textureSampleLevel(color, state, vec2f(0.5), 0.0)",
])
func rejectsInvalidMultisampledTextureOperations(call: String) throws {
    let source = """
        @group(0) @binding(0) var color: texture_multisampled_2d<f32>;
        @group(0) @binding(1) var depth: texture_depth_multisampled_2d;
        @group(0) @binding(2) var single: texture_2d<f32>;
        @group(0) @binding(3) var state: sampler;
        @compute @workgroup_size(1) fn run() { let value = \(call); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid multisampled texture operation accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

@Test func validatesMultisampledTextureDescriptors() throws {
    let capabilities = try DeviceCapabilities(maxBufferSize: 1024)
    let valid = TextureDescriptor(width: 2, height: 2, format: .rgba8Unorm,
                                  usage: [.renderAttachment, .textureBinding], sampleCount: 4)
    try valid.validate(against: capabilities)
    for invalid in [
        TextureDescriptor(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment, sampleCount: 2),
        .init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment, mipLevelCount: 2, sampleCount: 4),
        .init(width: 2, height: 2, format: .rgba8Unorm, usage: .textureBinding, sampleCount: 4),
        .init(width: 2, height: 2, format: .rgba8Unorm, usage: [.renderAttachment, .storageBinding], sampleCount: 4),
    ] {
        #expect(throws: GPUError.self) { try invalid.validate(against: capabilities) }
    }
}

@Test func validatesColorResolveAttachmentsAndSubresourceAliasing() throws {
    let source = ResourceID(device: 1, serial: 1)
    let destination = ResourceID(device: 1, serial: 2)
    let view = ResourceID(device: 1, serial: 3)
    let sibling = ResourceID(device: 1, serial: 4)
    let slot = ComputeBindingSlot(binding: 0)
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let multisampled = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm,
                                          usage: .renderAttachment, sampleCount: 4)
    let valid = TextureDescriptor(width: 2, height: 2, format: .rgba8Unorm,
                                  usage: [.renderAttachment, .textureBinding], mipLevelCount: 2)
    let pipeline = RenderPipeline(id: .init(device: 1, serial: 5), colorFormat: .rgba8Unorm, sampleCount: 4)
    func command(_ target: RenderResolveAttachment) -> BufferCommand {
        .render(pipeline: pipeline, target: source, clear: clear, vertexCount: 0, resolve: target)
    }

    let resolve = command(.init(texture: destination, mipLevel: 1))
    #expect(resolve.resources.contains(destination))
    try resolve.validate(device: 1, descriptors: [:], textures: [source: multisampled, destination: valid])
    for invalid in [
        TextureDescriptor(width: 2, height: 2, format: .rgba8Unorm, usage: .textureBinding, mipLevelCount: 2),
        .init(width: 2, height: 2, format: .bgra8Unorm, usage: .renderAttachment, mipLevelCount: 2),
        .init(width: 4, height: 4, format: .rgba8Unorm, usage: .renderAttachment, mipLevelCount: 2),
        .init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment),
    ] {
        #expect(throws: GPUError.self) {
            try resolve.validate(device: 1, descriptors: [:], textures: [source: multisampled, destination: invalid])
        }
    }
    #expect(throws: GPUError.self) {
        try command(.init(texture: destination)).validate(
            device: 1, descriptors: [:], textures: [source: multisampled, destination: multisampled])
    }
    #expect(throws: GPUError.self) {
        let singleSample = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment)
        let singlePipeline = RenderPipeline(id: pipeline.id, colorFormat: .rgba8Unorm)
        try BufferCommand.render(pipeline: singlePipeline, target: source, clear: clear, vertexCount: 0,
                                 resolve: .init(texture: destination, mipLevel: 1)).validate(
            device: 1, descriptors: [:], textures: [source: singleSample, destination: valid])
    }

    let sampling = RenderPipeline(id: pipeline.id, colorFormat: .rgba8Unorm, sampleCount: 4, sampledTextures: [
        .init(handle: .init(slot: slot, vertexIndex: nil, fragmentIndex: 0, visibility: 2),
              requirement: .init(slot: slot, sampleType: .float)),
    ])
    let views: [ResourceID: TextureViewResource] = [
        view: .init(texture: destination, levels: 1..<2),
        sibling: .init(texture: destination, levels: 0..<1),
    ]
    func validate(_ sampled: ResourceID, resolve: RenderResolveAttachment = .init(texture: view),
                  textureViews: [ResourceID: TextureViewResource] = views) throws {
        try BufferCommand.render(pipeline: sampling, target: source, clear: clear, vertexCount: 0,
                                 textures: [slot: sampled], resolve: resolve)
            .validate(device: 1, descriptors: [:], textures: [source: multisampled, destination: valid],
                      samplerDescriptors: [:], textureViews: textureViews)
    }
    try validate(sibling)
    #expect(throws: GPUError.self) { try validate(view) }
    #expect(throws: GPUError.self) { try validate(destination) }
    #expect(throws: GPUError.self) { try validate(sibling, resolve: .init(texture: view, mipLevel: 1)) }
    #expect(throws: GPUError.self) {
        var invalidViews = views
        invalidViews[view] = .init(texture: destination, levels: 0..<2)
        try validate(sibling, textureViews: invalidViews)
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func resolvesColorAndPreservesIndividualMultisampleValuesOnMetal(indexed: Bool) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let stages = ShaderInput.wgsl("""
            @vertex fn vs(@location(0) xy: vec2f) -> @builtin(position) vec4f { return vec4f(xy, 0.25, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
            """)
        let render = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(stages, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(stages, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm,
            vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])],
            depthState: .init(compare: .less), sampleCount: 4)
        let compute = try await device.createComputePipeline(
            compiler.compileCompute(.wgsl(multisampleReader), to: .msl))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .textureBinding, .copySource], sampleCount: 4))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding, .copySource], sampleCount: 4))
        let resolved = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource], mipLevelCount: 2))
        let resolvedView = try await device.createTextureView(texture: resolved, .init(baseMipLevel: 1))
        let colorCopy = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .textureBinding, .copyDestination], sampleCount: 4))
        let depthCopy = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding, .copyDestination], sampleCount: 4))
        let colorView = try await device.createTextureView(texture: colorCopy)
        let single = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .textureBinding]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
        let out = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
        func draw(_ target: ResourceID) -> BufferCommand {
            let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            if indexed {
                return .renderIndexed(pipeline: render, target: target, clear: clear,
                                      indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3),
                                      vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth),
                                      resolve: .init(texture: resolvedView))
            }
            return .render(pipeline: render, target: target, clear: clear, vertexCount: 3,
                           vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth),
                           resolve: .init(texture: resolved, mipLevel: 1))
        }
        func load(_ target: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): target, .init(binding: 1): depthCopy])
        }
        await #expect(throws: GPUError.self) { _ = try await device.submitCommands([draw(single)]) }
        await #expect(throws: GPUError.self) { _ = try await device.submitCommands([load(single)]) }
        await #expect(throws: GPUError.self) {
            _ = try await device.submitCommands([.copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [Float(-3), -1, 0, -1, 0, 5].map(\.bitPattern)),
            .write(buffer: indices, offset: 0, words: [0, 1, 2]),
            draw(color),
            .copyTextureToTexture(source: color, destination: colorCopy, sourceRegion: .init(width: 1, height: 1)),
            .copyTextureToTexture(source: depth, destination: depthCopy, sourceRegion: .init(width: 1, height: 1)),
            load(colorView),
            .copy(source: out, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
            .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 4),
            .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 3)
        let words = snapshots[0]
        try #require(words.count == 12)
        #expect(words[8..<12].elementsEqual([4, 4, 2, 0]))
        let coverage = words[0..<4]
        #expect(coverage.sorted() == [0, 0, 1, 1])
        #expect(snapshots[1] == [0xff000080])
        #expect(snapshots[2] == [0, 0])
        #expect(zip(coverage, words[4..<8]).allSatisfy { color, depth in
            color <= 1 && depth == (color == 1 ? 1 : 4)
        })
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
