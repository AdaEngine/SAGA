import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let storeActionShader = ShaderInput.wgsl("""
    @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
    @fragment fn fs() {}
    @group(0) @binding(0) var depth: texture_depth_2d;
    @group(0) @binding(1) var stencil: texture_2d<u32>;
    @group(0) @binding(2) var<storage, read_write> out: vec2u;
    @compute @workgroup_size(1) fn readAspects() {
        out = vec2u(bitcast<u32>(textureLoad(depth, vec2u(0), 0)), textureLoad(stencil, vec2u(0), 0).r);
    }
    """)

#if canImport(Metal)
@Test func renderStoreActionDiscardsColorAfterResolveOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(storeActionShader, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(storeActionShader, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment,
            colorFormat: .rgba8Unorm, sampleCount: 4, additionalColorTargets: [2: .init(format: .rgba8Unorm)])
        let msaa = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment], sampleCount: 4)
        let single = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource])
        let primary = try await device.createTexture(msaa)
        let secondary = try await device.createTexture(msaa)
        let resolved = try await device.createTexture(single)
        let kept = try await device.createTexture(single)
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let red = RenderClearColor(red: 1, green: 0, blue: 0, alpha: 1)
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: primary, clear: red, vertexCount: 0, resolve: .init(texture: resolved),
                    additionalColors: [2: .init(texture: secondary, loadAction: .clear(blue))], colorStore: .discard),
            .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .render(pipeline: pipeline, target: primary, colorLoad: .load, vertexCount: 0, resolve: .init(texture: resolved),
                    additionalColors: [2: .init(texture: secondary, resolve: .init(texture: kept), storeAction: .discard)]),
            .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .copyTextureToBuffer(source: kept, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .render(pipeline: pipeline, target: primary, colorLoad: .load, vertexCount: 0,
                    additionalColors: [2: .init(texture: secondary, resolve: .init(texture: kept))]),
            .copyTextureToBuffer(source: kept, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        ])
        #expect(try await ticket.wait() == [[0xff0000ff], [0], [0xffff0000], [0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func renderStoreActionDiscardsDepthAndStencilIndependentlyOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(storeActionShader, entryPoint: "vs", to: .msl)
    let shader = try compiler.compileCompute(storeActionShader, entryPoint: "readAspects", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: nil, colorFormat: nil,
            depthState: .init(format: .depth24PlusStencil8, compare: .always, writeEnabled: false))
        let depthStencil = try await device.createTexture(.init(width: 1, height: 1, format: .depth24PlusStencil8,
            usage: [.renderAttachment, .textureBinding]))
        let depth = try await device.createTextureView(texture: depthStencil, .init(aspect: .depthOnly))
        let stencil = try await device.createTextureView(texture: depthStencil, .init(aspect: .stencilOnly))
        let compute = try await device.createComputePipeline(shader)
        let out = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let readAspects = BufferCommand.dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out],
            workgroups: .init(x: 1), textures: [.init(binding: 0): depth, .init(binding: 1): stencil])
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 0,
                    depth: .init(texture: depthStencil, loadAction: .clear(0.5),
                                 stencil: .init(loadAction: .clear(17)), storeAction: .discard)),
            readAspects, .copy(source: out, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
            .render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 0,
                    depth: .init(texture: depthStencil, loadAction: .clear(0.5),
                                 stencil: .init(loadAction: .load, storeAction: .discard))),
            readAspects, .copy(source: out, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
        ])
        #expect(try await ticket.wait() == [[0, 17], [Float(0.5).bitPattern, 0]])
        await #expect(throws: GPUError.validation("Read-only or unused aspects cannot be discarded")) {
            try await device.submitCommands([.render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 0,
                depth: .init(texture: depthStencil, loadAction: .readOnly, stencil: .init(loadAction: .readOnly),
                             storeAction: .discard))])
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
