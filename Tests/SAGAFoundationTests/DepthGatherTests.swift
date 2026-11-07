import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private let depthGatherShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
    @group(0) @binding(1) var image: texture_depth_2d;
    @group(0) @binding(2) var state: sampler_comparison;
    fn gather(uv: vec2f) -> vec4f { return textureGatherCompare(image, state, uv, 0.375); }
    @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
        if id.x == 0u {
            let uv = vec2f(0.33333334, 0.5);
            out[0u] = gather(uv);
            out[1u] = textureGatherCompare(image, state, uv, 0.625, vec2i(1, 0));
            out[2u] = textureGatherCompare(image, state, uv, 0.375, vec2i(-1, 0));
            out[3u] = textureGatherCompare(image, state, uv, 0.25);
        }
    }
    """

private let rawDepthGatherShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
    @group(0) @binding(1) var image: texture_depth_2d;
    @group(0) @binding(2) var state: sampler;
    fn gather(uv: vec2f) -> vec4f { return textureGather(image, state, uv); }
    @compute @workgroup_size(1) fn run() {
        let uv = vec2f(0.33333334, 0.5);
        out[0u] = gather(uv);
        let depthView = image;
        out[1u] = textureGather((depthView), state, uv, vec2i(1, 0));
        out[2u] = textureGather(image, state, uv, vec2i(-1, 0));
        out[3u] = textureGather(image, state, uv);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsRawDepthGatherAcrossTargets(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(rawDepthGatherShader), to: format)
    #expect(shader.sampledTextures.first?.isDepth == true)
    #expect(shader.samplerBindings.first?.isComparison == false)
    #expect(shader.samplerBindings.first?.requiresNonFiltering == true)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("textureGather("))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl:
        #expect(body.contains(".gather(") && body.contains("floor("))
    case .hlsl:
        #expect(body.contains(".Gather(") && body.contains("int2(1, 0)"))
    case .glsl:
        #expect(body.contains("uniform sampler2D saga_pair_1_2"))
        #expect(body.contains("textureGatherOffset(saga_pair_1_2"))
    default: Issue.record("Unexpected target")
    }
}

@Test func renderStagesCanShareNearestSamplerForDepthAndColor() throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_depth_2d;
        @group(0) @binding(1) var color: texture_2d<f32>;
        @group(0) @binding(2) var state: sampler;
        @vertex fn vs() -> @builtin(position) vec4f {
            let depths = textureGather(image, state, vec2f(0.5));
            return vec4f(0.0, 0.0, depths.x, 1.0);
        }
        @fragment fn fs() -> @location(0) vec4f {
            return textureSample(color, state, vec2f(0.5));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    #expect(vertex.samplerBindings.first?.requiresNonFiltering == true)
    #expect(fragment.samplerBindings.first?.requiresNonFiltering == false)
    let pipeline = try RenderPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm,
                                      vertexBuffers: [], vertex: vertex, fragment: fragment)
    #expect(pipeline.samplerBindings.first?.requirement.requiresNonFiltering == true)
}

private let depthGridShader = """
    @vertex fn vs(@location(0) point: vec3f) -> @builtin(position) vec4f { return vec4f(point, 1.0); }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsDepthGatherComparisonAcrossTargets(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(depthGatherShader), to: format)
    #expect(shader.sampledTextures.first?.isDepth == true)
    #expect(shader.samplerBindings.first?.isComparison == true)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("textureGatherCompare("))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl:
        #expect(body.contains(".gather_compare(") && body.contains("floor("))
    case .hlsl:
        #expect(body.contains(".GatherCmp(") && body.contains("int2(1, 0)"))
        #expect(body.contains("0.375f, int2(0, 0)"))
    case .glsl:
        #expect(body.contains("sampler2DShadow"))
        #expect(body.contains("textureGatherOffset(saga_pair_1_2"))
        #expect(body.contains("0.625, ivec2(1, 0)"))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    ("textureGather(image, comparison, vec2f(0.5))", ShaderDiagnosticCode.typeMismatch),
    ("textureGather(image, state, vec2f(0.5), vec2i(0, -9))", ShaderDiagnosticCode.invalidType),
    ("textureGatherCompare(image, state, vec2f(0.5), 0.5)", ShaderDiagnosticCode.typeMismatch),
    ("textureGatherCompare(color, comparison, vec2f(0.5), 0.5)", ShaderDiagnosticCode.typeMismatch),
    ("textureGatherCompare(image, comparison, vec2f(0.5), 1u)", ShaderDiagnosticCode.typeMismatch),
    ("textureGatherCompare(image, comparison, vec2f(0.5), 0.5, vec2i(8, 0))", ShaderDiagnosticCode.invalidType),
])
func rejectsInvalidDepthGatherOperands(call: String, diagnostic: ShaderDiagnosticCode) throws {
    let source = """
        @group(0) @binding(0) var image: texture_depth_2d;
        @group(0) @binding(1) var color: texture_2d<f32>;
        @group(0) @binding(2) var state: sampler;
        @group(0) @binding(3) var comparison: sampler_comparison;
        @compute @workgroup_size(1) fn run() { let gathered = \(call); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid depth gather accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == diagnostic)
    }
}

#if canImport(Metal)
@Test(arguments: [true, false])
func gathersRenderedDepthInWGSLOrderOnMetal(comparison: Bool) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(depthGridShader)
        let render = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm,
            vertexBuffers: [.init(stride: 12, attributes: [.init(location: 0, offset: 0, format: .float32x3)])],
            depthState: .init())
        let compute = try await device.createComputePipeline(
            compiler.compileCompute(.wgsl(comparison ? depthGatherShader : rawDepthGatherShader), to: .msl))
        let depth = try await device.createTexture(.init(width: 6, height: 4, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding], mipLevelCount: 2))
        let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
        let color = try await device.createTexture(.init(width: 3, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
        let sampler = try await device.createSampler(.init(addressModeU: .repeat, compare: comparison ? .less : nil))
        let vertices = try await device.createBuffer(.init(size: 432, usage: [.vertex, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let x: [Float] = [-1, -1 / 3, 1 / 3, 1]
        let y: [Float] = [1, 0, -1]
        let values: [[Float]] = [[0.125, 0.375, 0.625], [0.25, 0.5, 0.875]]
        var grid: [Float] = []
        for row in 0..<2 {
            for column in 0..<3 {
                let (l, r, t, b, z) = (x[column], x[column + 1], y[row], y[row + 1], values[row][column])
                grid += [l, t, z, l, b, z, r, t, z, r, t, z, l, b, z, r, b, z]
            }
        }
        if !comparison {
            let linear = try await device.createSampler(.init(minFilter: .linear))
            await #expect(throws: GPUError.validation("Invalid sampler binding")) {
                _ = try await device.submitCommands([
                    .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                                     textures: [.init(binding: 1): view], samplers: [.init(binding: 2): linear]),
                ])
            }
        }
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: grid.map(\.bitPattern)),
            .render(pipeline: render, target: color, clear: .init(red: 0, green: 0, blue: 0, alpha: 1),
                    vertexCount: 36, vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: view)),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 1): view], samplers: [.init(binding: 2): sampler]),
            .copy(source: output, destination: read, byteCount: 64),
            .read(buffer: read, byteCount: 64),
        ])
        let expected: [Float] = comparison
            ? [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
            : [0.25, 0.5, 0.375, 0.125, 0.5, 0.875, 0.625, 0.375,
               0.875, 0.25, 0.125, 0.625, 0.25, 0.5, 0.375, 0.125]
        #expect(try await ticket.wait() == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
