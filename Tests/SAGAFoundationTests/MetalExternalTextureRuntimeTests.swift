#if canImport(Metal)
import Testing
import SAGACore
import SAGATypes
import SAGAShaderIR
import SAGACompiler
import SAGAMetal

@Test func metalPlanarExternalTexturesBindAndRetainPlanesThroughDevice() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var rgba: texture_external;
        @group(0) @binding(1) var interleaved: texture_external;
        @group(0) @binding(2) var separate: texture_external;
        @group(0) @binding(3) var state: sampler;
        @group(0) @binding(4) var<storage, read_write> output: array<vec4f, 3>;
        @compute @workgroup_size(1) fn run() {
            output[0] = textureSampleBaseClampToEdge(rgba, state, vec2f(-1.0, 0.5));
            output[1] = textureLoad(interleaved, vec2u(0));
            output[2] = textureLoad(separate, vec2u(1, 0));
        }
        """), to: .msl))
    let graphics = ShaderInput.wgsl("""
        @group(0) @binding(0) var frame: texture_external;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            let xy = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return vec4f(xy[i], 0, 1);
        }
        @fragment fn fs() -> @location(0) vec4f { return textureLoad(frame, vec2i(0)); }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(graphics, entryPoint: "vs", to: .msl),
        fragment: compiler.compileRenderStage(graphics, entryPoint: "fs", to: .msl), colorFormat: .rgba8Unorm)
    let usage: TextureUsage = [.textureBinding, .copyDestination]
    let rgba = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                    usage: usage.union(.renderAttachment)))
    let y = try await device.createTexture(.init(width: 2, height: 1, format: .r8Unorm, usage: usage))
    let uv = try await device.createTexture(.init(width: 1, height: 1, format: .rg8Unorm, usage: usage))
    let cb = try await device.createTexture(.init(width: 1, height: 1, format: .r8Unorm, usage: usage))
    let cr = try await device.createTexture(.init(width: 1, height: 1, format: .r8Unorm, usage: usage))
    let rgbaView = try await device.createTextureView(texture: rgba)
    let cropped = try await device.importExternalTexture(.init(format: .rgba, planes: [rgbaView], width: 1, height: 1,
        sampleTransform: [0.5, 0, 0, 1, 0.5, 0], loadTransform: [1, 0, 0, 1, 1, 0]))
    let interleaved = try await device.importExternalTexture(.init(format: .nv12, planes: [y, uv]))
    let separate = try await device.importExternalTexture(.init(format: .yu12, planes: [y, cb, cr]))
    await #expect(throws: GPUError.validation("Invalid external texture plane format")) {
        try await device.importExternalTexture(.init(format: .nv12, planes: [rgbaView, uv]))
    }
    await #expect(throws: GPUError.validation("External texture transform must preserve axis alignment")) {
        try await device.importExternalTexture(.init(format: .rgba, planes: [rgbaView],
            sampleTransform: [1, 0, 1, 1, 0, 0]))
    }
    await #expect(throws: GPUError.validation("External texture transforms exceed the plane or logical extent")) {
        try await device.importExternalTexture(.init(format: .rgba, planes: [rgbaView], width: 1, height: 1))
    }
    await #expect(throws: GPUError.validation("Render attachment cannot be bound as an external texture plane")) {
        try await device.submitCommands([
            .render(pipeline: render, target: rgba, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 0): cropped]),
        ])
    }
    let sampler = try await device.createSampler(.init(addressModeU: .repeat))
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                     usage: [.renderAttachment, .copySource]))
    let dispatch = BufferCommand.dispatchCompute(pipeline: compute, bindings: [.init(binding: 4): output],
        workgroups: .init(x: 1), textures: [.init(binding: 0): cropped, .init(binding: 1): interleaved, .init(binding: 2): separate],
        samplers: [.init(binding: 3): sampler])
    let ticket = try await device.submitCommands([
        .writeTexture(destination: rgba, data: [255, 0, 0, 255, 0, 255, 0, 255], region: .init(width: 2, height: 1)),
        .writeTexture(destination: y, data: [64, 96], region: .init(width: 2, height: 1)),
        .writeTexture(destination: uv, data: [128, 192], region: .init(width: 1, height: 1)),
        .writeTexture(destination: cb, data: [128], region: .init(width: 1, height: 1)),
        .writeTexture(destination: cr, data: [192], region: .init(width: 1, height: 1)),
        dispatch,
        .copy(source: output, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
        .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3, textures: [.init(binding: 0): cropped]),
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ])
    try await device.destroyTexture(rgba)
    let words = try await ticket.wait()
    let computeWords = try #require(words.first)
    #expect(computeWords.count == 12)
    let expected: [Float] = [0, 1, 0, 1, 64.0/255, 128.0/255, 192.0/255, 1, 96.0/255, 128.0/255, 192.0/255, 1]
    #expect(zip(computeWords.map(Float.init(bitPattern:)), expected).allSatisfy { abs($0.0 - $0.1) < 0.001 })
    #expect(words.last == [0xff00ff00])
    await #expect(throws: GPUError.self) { try await device.submitCommands([dispatch]) }
    try await device.destroyExternalTexture(cropped)
    try await device.destroyExternalTexture(interleaved)
    try await device.destroyExternalTexture(separate)
    await device.close()
}
#endif
