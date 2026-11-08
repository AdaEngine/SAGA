import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private let stencilOnlyRenderShader = """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    @fragment fn writesDepth() -> @builtin(frag_depth) f32 { return 0.5; }
    """

private let stencilOnlyReadShader = """
    @group(0) @binding(0) var allAspects: texture_2d<u32>;
    @group(0) @binding(1) var stencilView: texture_2d<u32>;
    @group(0) @binding(2) var<storage, read_write> out: vec4u;
    @compute @workgroup_size(1) fn run() {
        let a = vec2u(0, 0);
        let b = vec2u(1, 0);
        out = vec4u(textureLoad(allAspects, a, 0u).x, textureLoad(allAspects, b, 0u).x,
                    textureLoad(stencilView, a, 0u).x, textureLoad(stencilView, b, 0u).x);
    }
    """

@Test func stencil8HasAnEightBitAspectAndAlignedFullCopies() throws {
    let id = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let texture = TextureDescriptor(width: 2, height: 1, format: .stencil8,
                                    usage: [.copySource, .copyDestination, .textureBinding, .renderAttachment])
    try texture.validate(against: .init(maxBufferSize: 64))
    let buffers = [buffer: BufferDescriptor(size: 8, usage: [.copySource, .copyDestination])]
    func check(_ command: BufferCommand) throws {
        try command.validate(device: 1, descriptors: buffers, textures: [id: texture])
    }
    let region = TextureCopyRegion(width: 2, height: 1)
    try check(.copyBufferToTexture(source: buffer, layout: .init(offset: 4), destination: id, region: region))
    try check(.copyTextureToBuffer(source: id, destination: buffer, bytesPerRow: 256, destinationOffset: 4))
    try check(.writeTexture(destination: id, data: [0xee, 3, 7], layout: .init(offset: 1), region: region, aspect: .stencilOnly))
    #expect(throws: GPUError.self) {
        try check(.copyTextureToBuffer(source: id, destination: buffer, bytesPerRow: 256, aspect: .depthOnly))
    }
    #expect(throws: GPUError.self) {
        try check(.copyBufferToTexture(source: buffer, layout: .init(offset: 1), destination: id, region: region))
    }
    #expect(throws: GPUError.self) {
        try check(.writeTexture(destination: id, data: [3], region: .init(width: 1, height: 1)))
    }
}

@Test func stencilOnlyPipelinesRejectDepthStateAndDepthOutputs() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(stencilOnlyRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let writesDepth = try compiler.compileRenderStage(source, entryPoint: "writesDepth", to: .msl)
    let id = ResourceID(device: 1, serial: 1)
    let stencilOnly = RenderDepthState(format: .stencil8, compare: .always, writeEnabled: false)
    let pipeline = try RenderPipeline(id: id, colorFormat: .rgba8Unorm, vertexBuffers: [],
                                      depthState: stencilOnly, vertex: vertex, fragment: fragment)
    #expect(throws: GPUError.self) {
        try RenderPipeline(id: id, colorFormat: .rgba8Unorm, vertexBuffers: [],
                           depthState: .init(format: .stencil8), vertex: vertex, fragment: fragment)
    }
    #expect(throws: GPUError.self) {
        try RenderPipeline(id: id, colorFormat: .rgba8Unorm, vertexBuffers: [],
                           depthState: stencilOnly, vertex: vertex, fragment: writesDepth)
    }
    let color = ResourceID(device: 1, serial: 2)
    let stencil = ResourceID(device: 1, serial: 3)
    let textures: [ResourceID: TextureDescriptor] = [
        color: .init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment),
        stencil: .init(width: 1, height: 1, format: .stencil8, usage: .renderAttachment),
    ]
    for action: RenderDepthAttachment.LoadAction in [.clear(1), .load, .readOnly, .unused] {
        let command = BufferCommand.render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                                           depth: .init(texture: stencil, loadAction: action, stencil: .init()))
        if action == .unused {
            try command.validate(device: 1, descriptors: [:], textures: textures)
        } else {
            #expect(throws: GPUError.self) { try command.validate(device: 1, descriptors: [:], textures: textures) }
        }
    }
}

#if canImport(Metal)
@Test func rendersCopiesAndSamplesStandaloneStencilOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let words = try await stencilOnlyDiagnostic(device)
    #expect(words == [0x0000, 0x0703, 0xff00ff00, 0, 0x0704, 4, 7, 4, 7])
    await device.close()
}

private func stencilOnlyDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let original = try await device.createTexture(.init(width: 2, height: 1, format: .stencil8,
        usage: [.renderAttachment, .copySource, .copyDestination]))
    let clone = try await device.createTexture(.init(width: 2, height: 1, format: .stencil8,
        usage: [.copySource, .copyDestination, .textureBinding]))
    let view = try await device.createTextureView(texture: clone, .init(aspect: .stencilOnly))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let source = ShaderInput.wgsl(stencilOnlyRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let face = RenderStencilFaceState(compare: .equal, pass: .incrementClamp)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        depthState: .init(format: .stencil8, compare: .always, writeEnabled: false, stencil: .init(front: face, back: face)))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(stencilOnlyReadShader), to: .msl))
    let region = TextureCopyRegion(width: 2, height: 1)
    return try await device.submitCommands([
        .copyTextureToBuffer(source: original, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .writeTexture(destination: original, data: [0xee, 3, 7], layout: .init(offset: 1), region: region),
        .copyTextureToBuffer(source: original, destination: readback, bytesPerRow: 256, aspect: .stencilOnly),
        .read(buffer: readback, byteCount: 4),
        .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: original, loadAction: .unused, stencil: .init(loadAction: .load, reference: 3))),
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        .copyTextureToTexture(source: original, destination: clone, sourceRegion: region),
        .write(buffer: readback, offset: 0, words: [0]),
        .copyTextureToBuffer(source: clone, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): clone, .init(binding: 1): view]),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]).wait().flatMap { $0 }
}
#endif
