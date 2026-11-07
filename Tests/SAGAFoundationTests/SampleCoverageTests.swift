import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let sampleCoverageDraw = """
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) marker: f32,
    }
    struct SampleInput {
        @builtin(sample_mask) coverage: u32,
        @location(0) marker: f32,
        @builtin(sample_index) index: u32,
    }
    struct SampleOutput {
        @builtin(sample_mask) coverage: u32,
        @location(0) color: vec4u,
    }
    @vertex fn vs(@location(0) xy: vec2f) -> VertexOutput {
        return VertexOutput(vec4f(xy, 0, 1), 1.0);
    }
    @fragment fn allSamples(@builtin(sample_index) index: u32) -> @location(0) vec4u {
        return vec4u(index + 1u, 0u, 0u, 1u);
    }
    @fragment fn maskedSamples(input: SampleInput) -> SampleOutput {
        let covered = (input.coverage & (1u << input.index)) != 0u && input.marker == 1.0;
        return SampleOutput(5u, vec4u(input.index + 1u, select(0u, 9u, covered), 0u, 1u));
    }
    """

private let sampleCoverageRead = """
    @group(0) @binding(0) var unmasked: texture_multisampled_2d<u32>;
    @group(0) @binding(1) var masked: texture_multisampled_2d<u32>;
    @group(0) @binding(2) var<storage, read_write> out: array<vec4u, 8>;
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        out[lane] = textureLoad(unmasked, vec2u(0), lane);
        out[lane + 4u] = textureLoad(masked, vec2u(0), lane);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsFragmentSampleIndexAndCoverageAcrossTargets(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(sampleCoverageDraw), entryPoint: "maskedSamples", to: format)
    #expect(shader.varyings.map(\.location) == [0])
    #expect(shader.colorOutputKind == .uint)
    #expect(shader.writesSampleMask)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("@builtin(sample_index)") && body.contains("@builtin(sample_mask)"))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl:
        #expect(body.contains("[[sample_id]]") && body.contains("[[sample_mask]]"))
    case .hlsl:
        #expect(body.contains("uint saga_sample_index : SV_SampleIndex;"))
        #expect(body.contains("uint saga_sample_mask : SV_Coverage;"))
        #expect(body.contains("saga_input.saga_sample_mask"))
    case .glsl:
        #expect(body.contains("uint(gl_SampleID)") && body.contains("uint(gl_SampleMaskIn[0])"))
        #expect(body.contains("gl_SampleMask[0] = int("))
    default: Issue.record("Unexpected target")
    }
}

#if canImport(Metal)
@Test func writesSampleIndicesAndMasksOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(sampleCoverageDraw)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let unmasked = try await device.createRenderPipeline(
            vertex: vertex, fragment: compiler.compileRenderStage(source, entryPoint: "allSamples", to: .msl),
            colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4, sampleMask: 0b1010)
        let masked = try await device.createRenderPipeline(
            vertex: vertex, fragment: compiler.compileRenderStage(source, entryPoint: "maskedSamples", to: .msl),
            colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4, sampleMask: 0b0110)
        let reader = try await device.createComputePipeline(compiler.compileCompute(.wgsl(sampleCoverageRead), to: .msl))
        let descriptor = TextureDescriptor(width: 1, height: 1, format: .rgba8Uint,
                                           usage: [.renderAttachment, .textureBinding], sampleCount: 4)
        let allSamples = try await device.createTexture(descriptor)
        let maskedSamples = try await device.createTexture(descriptor)
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let out = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            .render(pipeline: unmasked, target: allSamples, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)]),
            .render(pipeline: masked, target: maskedSamples, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)]),
            .dispatchCompute(pipeline: reader, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): allSamples, .init(binding: 1): maskedSamples]),
            .copy(source: out, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
        ])
        let expected: [UInt32] = [
            0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0, 1, 0, 0, 0, 0,
        ]
        #expect(try await ticket.wait() == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

private let alphaCoverageDraw = """
    struct FragmentOutput {
        @location(0) color: vec4f,
        @builtin(sample_mask) mask: u32,
    }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn zero() -> @location(0) vec4f { return vec4f(1, 0, 0, 0); }
    @fragment fn half() -> @location(0) vec4f { return vec4f(1, 0, 0, 0.5); }
    @fragment fn one() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
    @fragment fn masked() -> FragmentOutput { return FragmentOutput(vec4f(1), 5u); }
    @fragment fn integer() -> @location(0) vec4u { return vec4u(1); }
    @fragment fn depthOnly() -> @builtin(frag_depth) f32 { return 0.25; }
    """

private let alphaCoverageRead = """
    @group(0) @binding(0) var image: texture_multisampled_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> out: array<u32, 4>;
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        out[lane] = select(0u, 1u, textureLoad(image, vec2u(0), lane).r == 1.0);
    }
    """

@Test func validatesAlphaToCoverageBeforePipelineAllocation() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(alphaCoverageDraw)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let one = try compiler.compileRenderStage(source, entryPoint: "one", to: .msl)
    let masked = try compiler.compileRenderStage(source, entryPoint: "masked", to: .msl)
    let integer = try compiler.compileRenderStage(source, entryPoint: "integer", to: .msl)
    await #expect(throws: GPUError.validation("Alpha-to-coverage requires a multisampled pipeline")) {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: one, colorFormat: .rgba8Unorm,
                                                  alphaToCoverageEnabled: true)
    }
    await #expect(throws: GPUError.validation("Alpha-to-coverage is incompatible with fragment sample_mask output")) {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: masked, colorFormat: .rgba8Unorm,
                                                  sampleCount: 4, alphaToCoverageEnabled: true)
    }
    await #expect(throws: GPUError.validation("Alpha-to-coverage requires a blendable color attachment and a fragment alpha output")) {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: integer, colorFormat: .rgba8Uint,
                                                  sampleCount: 4, alphaToCoverageEnabled: true)
    }
    await device.close()
}

#if canImport(Metal)
@Test func alphaToCoverageCombinesWithPipelineMaskOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(alphaCoverageDraw)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        var pipelines: [RenderPipeline] = []
        for (entry, mask) in [("zero", UInt32.max), ("half", UInt32.max), ("one", 0b1010), ("zero", 0b1010)] {
            let fragment = try compiler.compileRenderStage(source, entryPoint: entry, to: .msl)
            pipelines.append(try await device.createRenderPipeline(
                vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                sampleCount: 4, sampleMask: mask, alphaToCoverageEnabled: true))
        }
        let reader = try await device.createComputePipeline(compiler.compileCompute(.wgsl(alphaCoverageRead), to: .msl))
        let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .textureBinding], sampleCount: 4))
        let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = []
        for pipeline in pipelines {
            commands += [
                .render(pipeline: pipeline, target: image, clear: .transparentBlack, vertexCount: 3),
                .dispatchCompute(pipeline: reader, bindings: [.init(binding: 1): out], workgroups: .init(x: 1),
                                 textures: [.init(binding: 0): image]),
                .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
            ]
        }
        let onlyDepth = try compiler.compileRenderStage(source, entryPoint: "depthOnly", to: .msl)
        let noSamples = try await device.createRenderPipeline(
            vertex: vertex, fragment: onlyDepth, colorFormat: .rgba8Unorm,
            depthState: .init(compare: .always), sampleMask: 0)
        let singleColor = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: .renderAttachment))
        let singleDepth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .copySource]))
        commands += [
            .render(pipeline: noSamples, target: singleColor, clear: .transparentBlack, vertexCount: 3,
                    depth: .init(texture: singleDepth, loadAction: .clear(0.75))),
            .copyTextureToBuffer(source: singleDepth, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 4),
        ]
        let ticket = try await device.submitCommands(commands)
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 5)
        #expect(snapshots[0] == [0, 0, 0, 0])
        #expect(snapshots[1].contains(0) && snapshots[1].contains(1))
        #expect(snapshots[2] == [0, 1, 0, 1])
        #expect(snapshots[3] == [0, 0, 0, 0])
        #expect(snapshots[4] == [Float(0.75).bitPattern])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
