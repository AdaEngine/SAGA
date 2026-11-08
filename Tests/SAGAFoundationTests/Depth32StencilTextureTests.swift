import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

@Test func depth32StencilCopiesRequireOneAspectAndDirection() throws {
    let id = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let texture = TextureDescriptor(width: 2, height: 1, format: .depth32FloatStencil8,
                                    usage: [.copySource, .copyDestination, .renderAttachment, .textureBinding])
    let core = try DeviceCapabilities(maxBufferSize: 64)
    let enabled = try DeviceCapabilities(maxBufferSize: 64, features: [.depth32FloatStencil8])
    #expect(throws: GPUError.validation("Feature not advertised")) { try texture.validate(against: core) }
    try texture.validate(against: enabled)
    let descriptors = [buffer: BufferDescriptor(size: 16, usage: [.copySource, .copyDestination])]
    func check(_ command: BufferCommand) throws {
        try command.validate(device: 1, descriptors: descriptors, textures: [id: texture])
    }
    let region = TextureCopyRegion(width: 2, height: 1)
    try check(.copyTextureToBuffer(source: id, destination: buffer, bytesPerRow: 256, aspect: .depthOnly, destinationOffset: 4))
    try check(.copyTextureToBuffer(source: id, destination: buffer, bytesPerRow: 256, aspect: .stencilOnly, destinationOffset: 4))
    try check(.copyBufferToTexture(source: buffer, layout: .init(offset: 4), destination: id, region: region, aspect: .stencilOnly))
    try check(.writeTexture(destination: id, data: [0xee, 3, 7], layout: .init(offset: 1), region: region, aspect: .stencilOnly))
    for command: BufferCommand in [
        .copyTextureToBuffer(source: id, destination: buffer, bytesPerRow: 256),
        .copyTextureToBuffer(source: id, destination: buffer, bytesPerRow: 256, aspect: .depthOnly, destinationOffset: 2),
        .copyTextureToBuffer(source: id, destination: buffer, bytesPerRow: 256, aspect: .stencilOnly, destinationOffset: 2),
        .copyBufferToTexture(source: buffer, layout: .init(), destination: id, region: region, aspect: .depthOnly),
        .writeTexture(destination: id, data: Array(repeating: 0, count: 8), region: region, aspect: .depthOnly),
        .writeTexture(destination: id, data: [3], region: .init(width: 1, height: 1), aspect: .stencilOnly),
    ] {
        #expect(throws: GPUError.self) { try check(command) }
    }
}

private let depth32StencilRenderShader = """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0.25, 1); }
        if i == 1u { return vec4f(3, -1, 0.25, 1); }
        return vec4f(-1, 3, 0.25, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    """

private let depth32StencilReadShader = """
    @group(0) @binding(0) var depth: texture_depth_2d;
    @group(0) @binding(1) var stencil: texture_2d<u32>;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = vec2u(0, 0);
        let b = vec2u(1, 0);
        out = vec4f(textureLoad(depth, a, 0u), textureLoad(depth, b, 0u),
                    f32(textureLoad(stencil, a, 0u).x), f32(textureLoad(stencil, b, 0u).x));
    }
    """

#if canImport(Metal)
@Test func copiesAndSamplesBothDepth32StencilAspectsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.depth32FloatStencil8])
    let words = try await depth32StencilDiagnostic(device)
    #expect(words == [0x0703, 0xff00ff00, 0, Float(0.25).bitPattern, Float(0.75).bitPattern, 0x0704] +
        [Float(0.25), 0.75, 4, 7].map(\.bitPattern))
    await device.close()
}

private func depth32StencilDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let original = try await device.createTexture(.init(width: 2, height: 1, format: .depth32FloatStencil8,
        usage: [.renderAttachment, .copySource, .copyDestination]))
    let clone = try await device.createTexture(.init(width: 2, height: 1, format: .depth32FloatStencil8,
        usage: [.copySource, .copyDestination, .textureBinding]))
    let depth = try await device.createTextureView(texture: clone, .init(aspect: .depthOnly))
    let stencil = try await device.createTextureView(texture: clone, .init(aspect: .stencilOnly))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let source = ShaderInput.wgsl(depth32StencilRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let face = RenderStencilFaceState(compare: .equal, pass: .incrementClamp)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth32FloatStencil8, compare: .less, stencil: .init(front: face, back: face)))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(depth32StencilReadShader), to: .msl))
    let region = TextureCopyRegion(width: 2, height: 1)
    return try await device.submitCommands([
        .writeTexture(destination: original, data: [0xee, 3, 7], layout: .init(offset: 1), region: region, aspect: .stencilOnly),
        .copyTextureToBuffer(source: original, destination: readback, bytesPerRow: 256, aspect: .stencilOnly),
        .read(buffer: readback, byteCount: 4),
        .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: original, loadAction: .clear(0.75), stencil: .init(loadAction: .load, reference: 3))),
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        .copyTextureToTexture(source: original, destination: clone, sourceRegion: region),
        .copyTextureToBuffer(source: clone, destination: readback, bytesPerRow: 256, aspect: .depthOnly),
        .read(buffer: readback, byteCount: 8),
        .write(buffer: readback, offset: 0, words: [0]),
        .copyTextureToBuffer(source: clone, destination: readback, bytesPerRow: 256, aspect: .stencilOnly),
        .read(buffer: readback, byteCount: 4),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
        .copy(source: out, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]).wait().flatMap { $0 }
}
#endif
