import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let depthOnlyPassShader = ShaderInput.wgsl("""
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return vec4f(points[i], 0.5, 1);
    }
    @fragment fn fs() -> @builtin(frag_depth) f32 { return 0.25; }
    @fragment fn empty() {}
    @fragment fn sparse() -> @location(2) vec4u { return vec4u(19, 37, 73, 255); }
    """)

@Test func depthOnlyPassRequiresAnAttachmentAndRejectsAColorTarget() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileWebGPURenderStage(depthOnlyPassShader, entryPoint: "vs")
    let empty = try compiler.compileWebGPURenderStage(depthOnlyPassShader, entryPoint: "empty")
    let fragment = try compiler.compileWebGPURenderStage(depthOnlyPassShader, entryPoint: "fs")
    do {
        await #expect(throws: GPUError.validation("Render pipeline requires at least one attachment")) {
            try await device.createRenderPipeline(vertex: vertex, fragment: empty, colorFormat: nil)
        }
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: nil,
            depthState: .init(format: .depth32Float, compare: .always, writeEnabled: true))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float, usage: [.renderAttachment]))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment]))
        await #expect(throws: GPUError.validation("Color attachments do not match the render pipeline")) {
            try await device.submitCommands([.render(pipeline: pipeline, target: color, colorLoad: .load,
                                                     vertexCount: 3, depth: .init(texture: depth))])
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func vertexOnlyPipelineValidatesAttachments() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileWebGPURenderStage(depthOnlyPassShader, entryPoint: "vs")
    await #expect(throws: GPUError.validation("Render pipeline requires at least one attachment")) {
        try await device.createRenderPipeline(vertex: vertex, fragment: nil, colorFormat: nil)
    }
    await #expect(throws: GPUError.validation("Color targets require a fragment stage")) {
        try await device.createRenderPipeline(vertex: vertex, fragment: nil, colorFormat: nil,
            depthState: .init(), additionalColorTargets: [2: .init(format: .rgba8Unorm)])
    }
    await #expect(throws: GPUError.validation("Alpha-to-coverage requires a blendable color attachment and a fragment alpha output")) {
        try await device.createRenderPipeline(vertex: vertex, fragment: nil, colorFormat: nil,
            depthState: .init(), sampleCount: 4, alphaToCoverageEnabled: true)
    }
    await device.close()
}

#if canImport(Metal)
@Test(arguments: [UInt32.max, 0])
func vertexOnlyPipelineWritesRasterDepthOnMetal(_ mask: UInt32) async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct VertexOut {
            @builtin(position) position: vec4f,
            @location(0) unused: vec2f,
        }
        @group(0) @binding(0) var<uniform> parameters: vec4f;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> VertexOut {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return VertexOut(vec4f(points[i], parameters.x, 1), points[i]);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: nil, colorFormat: nil,
            depthState: .init(format: .depth32Float, compare: .always, writeEnabled: true), sampleMask: mask)
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .copySource]))
        let parameters = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .write(buffer: parameters, offset: 0, words: [Float(0.5).bitPattern, 0, 0, 0]),
            .render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 3,
                    uniforms: [.init(binding: 0): .init(buffer: parameters)],
                    depth: .init(texture: depth, loadAction: .clear(0.75))),
            .copyTextureToBuffer(source: depth, destination: readback, bytesPerRow: 256, aspect: .depthOnly),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[Float(mask == 0 ? 0.75 : 0.5).bitPattern]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func vertexOnlyPipelineMasksIndividualDepthSamplesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(depthOnlyPassShader, entryPoint: "vs", to: .msl)
    let shader = try compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var depth: texture_depth_multisampled_2d;
        @group(0) @binding(1) var<storage, read_write> output: array<f32, 4>;
        @compute @workgroup_size(4) fn readDepth(@builtin(local_invocation_index) sample: u32) {
            output[sample] = textureLoad(depth, vec2i(0), sample);
        }
        """), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let draw = try await device.createRenderPipeline(vertex: vertex, fragment: nil, colorFormat: nil,
            depthState: .init(compare: .always), sampleCount: 4, sampleMask: 0b1010)
        let read = try await device.createComputePipeline(shader)
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding], sampleCount: 4))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: draw, target: nil, colorLoad: .load, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .clear(0.75))),
            .dispatchCompute(pipeline: read, bindings: [.init(binding: 1): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): depth]),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(result == [[Float(0.75), 0.5, 0.75, 0.5].map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(arguments: [0, 1, 2])
func depthOnlyPassWritesAndReadsDepthOnMetal(_ drawMode: Int) async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(depthOnlyPassShader, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(depthOnlyPassShader, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: nil,
            depthState: .init(format: .depth32Float, compare: .always, writeEnabled: true))
        #expect(pipeline.colorFormat == nil && pipeline.colorTargets.isEmpty)
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let arguments = try await device.createBuffer(.init(size: 16, usage: [.indirect, .copyDestination]))
        let attachment = RenderDepthAttachment(texture: depth, loadAction: .clear(0.75))
        let draw: BufferCommand
        switch drawMode {
        case 1:
            draw = .renderIndexed(pipeline: pipeline, target: nil, colorLoad: .load,
                indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3), depth: attachment)
        case 2:
            draw = .renderIndirect(pipeline: pipeline, target: nil, colorLoad: .load, indirectBuffer: arguments, depth: attachment)
        default:
            draw = .render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 3, depth: attachment)
        }
        let snapshots = try await device.submitCommands([
            .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
            .write(buffer: arguments, offset: 0, words: [3, 1, 0, 0]),
            draw,
            .copyTextureToBuffer(source: depth, destination: readback, bytesPerRow: 256, aspect: .depthOnly),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(snapshots == [[Float(0.25).bitPattern]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
@Test func depthOnlyPassSupportAlsoAllowsSparseColorWithoutLocationZeroOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(depthOnlyPassShader, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(depthOnlyPassShader, entryPoint: "sparse", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: nil,
                                                             additionalColorTargets: [2: .init(format: .rgba8Uint)])
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
            usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 3,
                    additionalColors: [2: .init(texture: target, loadAction: .clear(.transparentBlack))]),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff492513]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
