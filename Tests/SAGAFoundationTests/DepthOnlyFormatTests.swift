import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

@Test func depth16HasAlignedLinearCopiesAndDepth24PlusIsOpaque() throws {
    let depth16 = ResourceID(device: 1, serial: 1)
    let depth24 = ResourceID(device: 1, serial: 2)
    let clone = ResourceID(device: 1, serial: 3)
    let buffer = ResourceID(device: 1, serial: 4)
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding, .renderAttachment]
    let descriptors = [
        depth16: TextureDescriptor(width: 2, height: 1, format: .depth16Unorm, usage: usage),
        depth24: TextureDescriptor(width: 2, height: 1, format: .depth24Plus, usage: usage),
        clone: TextureDescriptor(width: 2, height: 1, format: .depth24Plus, usage: usage),
    ]
    let capabilities = try DeviceCapabilities(maxBufferSize: 32)
    for descriptor in descriptors.values { try descriptor.validate(against: capabilities) }
    let buffers = [buffer: BufferDescriptor(size: 8, usage: [.copyDestination, .copySource])]
    let full = TextureCopyRegion(width: 2, height: 1)
    func check(_ command: BufferCommand) throws {
        try command.validate(device: 1, descriptors: buffers, textures: descriptors)
    }
    try check(.copyTextureToBuffer(source: depth16, destination: buffer, bytesPerRow: 256, destinationOffset: 4))
    try check(.copyBufferToTexture(source: buffer, layout: .init(offset: 4), destination: depth16, region: full, aspect: .depthOnly))
    try check(.writeTexture(destination: depth16, data: [0x99, 0, 64, 255, 191], layout: .init(offset: 1), region: full))
    try check(.copyTextureToTexture(source: depth24, destination: clone, sourceRegion: full))
    for command: BufferCommand in [
        .copyTextureToBuffer(source: depth16, destination: buffer, bytesPerRow: 256, destinationOffset: 2),
        .copyBufferToTexture(source: buffer, layout: .init(offset: 2), destination: depth16, region: full),
        .copyBufferToTexture(source: buffer, layout: .init(), destination: depth16, region: .init(width: 1, height: 1)),
        .copyTextureToBuffer(source: depth24, destination: buffer, bytesPerRow: 256, aspect: .depthOnly),
        .copyBufferToTexture(source: buffer, layout: .init(), destination: depth24, region: full),
        .writeTexture(destination: depth24, data: [0, 0, 0, 0], region: full),
    ] {
        #expect(throws: GPUError.self) { try check(command) }
    }
}

private let depthOnlyRenderShader = """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0.25, 1); }
        if i == 1u { return vec4f(3, -1, 0.25, 1); }
        return vec4f(-1, 3, 0.25, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    """

private let depthOnlyReadShader = """
    @group(0) @binding(0) var quantized: texture_depth_2d;
    @group(0) @binding(1) var opaqueDepth: texture_depth_2d;
    @group(0) @binding(2) var compare: sampler_comparison;
    @group(0) @binding(3) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(quantized, vec2u(0, 0), 0u);
        let b = textureLoad(quantized, vec2u(1, 0), 0u);
        let c = textureLoad(opaqueDepth, vec2u(0, 0), 0u);
        let d = textureSampleCompareLevel(opaqueDepth, compare, vec2f(0.25, 0.5), 0.2);
        out = vec4f(a, b, c, d);
    }
    """

#if canImport(Metal)
@Test func rendersCopiesAndSamplesDepthOnlyFormatsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let words = try await depthOnlyFormatDiagnostic(device)
    #expect(Array(words.prefix(3)) == [0xbfff4000, 0xffff8000, 0x40004000])
    let floats = words.dropFirst(3).map(Float.init(bitPattern:))
    #expect(floats.count == 4)
    for (actual, expected) in zip(floats, [Float(0.25), 0.25, 0.25, 1]) {
        #expect(abs(actual - expected) < 0.0001)
    }
    await device.close()
}

private func depthOnlyFormatDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let usage: TextureUsage = [.renderAttachment, .copySource, .copyDestination, .textureBinding]
    let depth16 = try await device.createTexture(.init(width: 2, height: 1, format: .depth16Unorm, usage: usage))
    let depth24 = try await device.createTexture(.init(width: 2, height: 1, format: .depth24Plus, usage: usage))
    let opaqueCopy = try await device.createTexture(.init(width: 2, height: 1, format: .depth24Plus,
                                                           usage: [.copyDestination, .textureBinding]))
    let depthView = try await device.createTextureView(texture: opaqueCopy, .init(aspect: .depthOnly))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let upload = try await device.createBuffer(.init(size: 8, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let sampler = try await device.createSampler(.init(compare: .less))
    let source = ShaderInput.wgsl(depthOnlyRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let render16 = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth16Unorm, compare: .less))
    let render24 = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth24Plus, compare: .less))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(depthOnlyReadShader), to: .msl))
    let full = TextureCopyRegion(width: 2, height: 1)
    let raw16: [BufferCommand] = [
        .copyTextureToBuffer(source: depth16, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ]
    let commands: [BufferCommand] = [
        .writeTexture(destination: depth16, data: [0x99, 0, 64, 255, 191], layout: .init(offset: 1), region: full),
    ] + raw16 + [
        .write(buffer: upload, offset: 0, words: [0xdeadbeef, 0xffff8000]),
        .copyBufferToTexture(source: upload, layout: .init(offset: 4), destination: depth16, region: full, aspect: .depthOnly),
    ] + raw16 + [
        .render(pipeline: render16, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: depth16, loadAction: .load)),
    ] + raw16 + [
        .render(pipeline: render24, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: depth24, loadAction: .clear(0.8))),
        .copyTextureToTexture(source: depth24, destination: opaqueCopy, sourceRegion: full),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 3): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth16, .init(binding: 1): depthView], samplers: [.init(binding: 2): sampler]),
        .copy(source: out, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}
#endif
