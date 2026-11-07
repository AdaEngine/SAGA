import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let depthComparisonShader = """
    @group(2) @binding(1) var image: texture_depth_2d;
    @group(2) @binding(2) var state: sampler_comparison;
    @group(2) @binding(3) var nearest: sampler;
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point, point.x * 0.5 + 0.5, 1.0);
    }
    @fragment fn writeDepth() -> @location(0) vec4f { return vec4f(0, 0, 0, 1); }
    @fragment fn fs() -> @location(0) vec4f {
        let uv = vec2f(0.25, 0.5);
        let a = textureSampleCompare(image, state, uv, 0.5);
        let b = textureSampleCompareLevel(image, state, uv, 0.5, vec2i(1, 0));
        let c = textureSample(image, nearest, uv);
        let d = textureSample(image, nearest, uv, vec2i(1, 0));
        return vec4f(a, b, c, d);
    }
    """

private let depthComparisonCompute = """
    @group(2) @binding(1) var image: texture_depth_2d;
    @group(2) @binding(2) var state: sampler_comparison;
    @group(2) @binding(3) var nearest: sampler;
    @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 2>;
    @compute @workgroup_size(1) fn run() {
        let a = textureSampleCompareLevel(image, state, vec2f(0.25, 0.5), 0.5);
        let b = textureSampleCompareLevel(image, state, vec2f(0.75, 0.5), 0.5);
        out[0] = vec4f(a, b, textureLoad(image, vec2u(0), 0u), textureLoad(image, vec2u(1, 0), 0u));
        let uv = vec2f(0.25, 0.5);
        let lod = 1i;
        out[1] = vec4f(textureSampleLevel(image, nearest, uv, 0),
                       textureSampleLevel(image, nearest, uv, 0u, vec2i(1, 0)),
                       textureSampleLevel(image, nearest, uv, lod),
                       textureSampleLevel(image, nearest, uv, u32(lod)));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsDepthComparisonAndRawFetchWithSeparateGLSLBindings(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fragment = try compiler.compileRenderStage(.wgsl(depthComparisonShader), entryPoint: "fs", to: format)
    let compute = try compiler.compileCompute(.wgsl(depthComparisonCompute), to: format)
    #expect(fragment.samplerBindings.first?.isComparison == true)
    #expect(compute.samplerBindings.first?.isComparison == true)
    #expect(fragment.samplerBindings.last?.requiresNonFiltering == true)
    #expect(compute.samplerBindings.last?.requiresNonFiltering == true)
    guard case .text(_, let body) = fragment.artifact else { Issue.record("Expected source"); return }
    guard case .text(_, let cs) = compute.artifact else { Issue.record("Expected compute source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("sampler_comparison"))
        #expect(body.contains("textureSampleCompare(") && body.contains("textureSampleCompareLevel("))
        _ = try compiler.parseAndValidate(.wgsl(body))
        #expect(body.contains("textureSample("))
        #expect(cs.contains("textureSampleLevel("))
        _ = try compiler.parseAndValidate(.wgsl(cs))
    case .msl:
        #expect(body.contains(".sample_compare(") && body.contains("level(0.0f)"))
        #expect(body.contains(".sample(") && cs.contains("level(float("))
    case .hlsl:
        #expect(body.contains("SamplerComparisonState"))
        #expect(body.contains(".SampleCmp(") && body.contains(".SampleCmpLevelZero("))
        #expect(body.contains(".Sample(") && cs.contains(".SampleLevel("))
    case .glsl:
        #expect(body.contains("uniform sampler2D saga_texture_0"))
        #expect(body.contains("uniform sampler2DShadow saga_pair_0_1"))
        #expect(body.contains("uniform sampler2D saga_pair_0_2"))
        #expect(body.contains("textureOffset(saga_pair_0_2") && body.contains(").x;"))
        #expect(cs.contains("textureLodOffset(saga_pair_0_2"))
        #expect(body.contains("textureLodOffset("))
        #expect(fragment.glslTextureSamplerBindings.map(\.isComparison) == [true, false])
    default: Issue.record("Unexpected format")
    }
}

@Test func implicitDepthComparisonRequiresFragmentStage() throws {
    let source = """
        @group(0) @binding(0) var image: texture_depth_2d;
        @group(0) @binding(1) var state: sampler_comparison;
        fn compareDepth() -> f32 { return textureSampleCompare(image, state, vec2f(0.5), 0.5); }
        @compute @workgroup_size(1) fn run() { let value = compareDepth(); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Implicit comparison accepted in compute")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .invalidControlFlow)
    }
}

@Test(arguments: [
    ("@compute @workgroup_size(1) fn run() { let result = readDepth(); }", ShaderDiagnosticCode.invalidControlFlow),
    ("""
    @fragment fn run(@location(0) uv: vec2f) -> @location(0) f32 {
        if uv.x > 0.5 { return readDepth(); }
        return 0.0;
    }
    """, ShaderDiagnosticCode.nonUniformControlFlow),
    ("""
    @compute @workgroup_size(1) fn run() { let result = textureSampleLevel(image, state, vec2f(0.5), 0.0); }
    """, ShaderDiagnosticCode.typeMismatch),
    ("""
    @fragment fn run() -> @location(0) f32 { return textureSampleBias(image, state, vec2f(0.5), 0.0); }
    """, ShaderDiagnosticCode.typeMismatch),
])
func validatesRawDepthSampling(entry: String, code: ShaderDiagnosticCode) throws {
    let source = """
        @group(0) @binding(0) var image: texture_depth_2d;
        @group(0) @binding(1) var state: sampler;
        fn readDepth() -> f32 { return textureSample(image, state, vec2f(0.5)); }
        \(entry)
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid raw depth sampling accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == code)
    }
}

#if canImport(Metal)
@Test func comparesRenderedDepthInFragmentAndComputeOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(depthComparisonShader)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let writer = try compiler.compileRenderStage(source, entryPoint: "writeDepth", to: .msl)
        let reader = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        let render = try await device.createRenderPipeline(
            vertex: vertex, fragment: writer, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
            depthState: .init(compare: .less))
        let preview = try await device.createRenderPipeline(
            vertex: vertex, fragment: reader, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
        let compute = try await device.createComputePipeline(
            compiler.compileCompute(.wgsl(depthComparisonCompute), to: .msl))
        let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding], mipLevelCount: 3))
        let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
        let compare = try await device.createSampler(.init(compare: .less))
        let plain = try await device.createSampler()
        let linear = try await device.createSampler(.init(magFilter: .linear))
        let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource], mipLevelCount: 2))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        let textures: [ComputeBindingSlot: ResourceID] = [.init(group: 2, binding: 1): view]
        let samplerSlot = ComputeBindingSlot(group: 2, binding: 2)
        let rawSamplerSlot = ComputeBindingSlot(group: 2, binding: 3)
        let streams = [VertexBufferBinding(buffer: vertices)]
        func visualize(_ sampler: ResourceID, raw: ResourceID) -> BufferCommand {
            .render(pipeline: preview, target: color, clear: clear, vertexCount: 3,
                    vertexBuffers: streams, textures: textures, samplers: [samplerSlot: sampler, rawSamplerSlot: raw])
        }
        await #expect(throws: GPUError.self) { try await device.submitCommands([visualize(plain, raw: plain)]) }
        await #expect(throws: GPUError.validation("Invalid sampler binding")) {
            _ = try await device.submitCommands([visualize(compare, raw: linear)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
            .render(pipeline: render, target: color, clear: clear, vertexCount: 3, vertexBuffers: streams,
                    depth: .init(texture: depth, mipLevel: 1)),
            .render(pipeline: render, target: color, clear: clear, vertexCount: 3, vertexBuffers: streams,
                    targetMipLevel: 1, depth: .init(texture: depth, mipLevel: 2)),
            visualize(compare, raw: plain),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                             textures: textures, samplers: [samplerSlot: compare, rawSamplerSlot: plain]),
            .copy(source: output, destination: read, byteCount: 32),
            .read(buffer: read, byteCount: 32),
        ])
        let compared: [Float] = [0, 1, 0.25, 0.75, 0.25, 0.75, 0.5, 0.5]
        let expected: [[UInt32]] = [[0xbf40ff00, 0xbf40ff00], compared.map(\.bitPattern)]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
