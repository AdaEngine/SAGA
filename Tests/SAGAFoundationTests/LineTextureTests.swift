import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let lineTextureShader = """
    @group(0) @binding(0) var line: texture_1d<f32>;
    @group(0) @binding(1) var nearest: sampler;
    @group(0) @binding(2) var readLine: texture_storage_1d<r32sint, read>;
    @group(0) @binding(3) var writeLine: texture_storage_1d<r32sint, read_write>;
    @group(0) @binding(4) var<storage, read_write> result: array<vec4f, 4>;
    @compute @workgroup_size(1) fn run() {
        result[0] = textureLoad(line, 0i, 0i);
        result[1] = textureSampleLevel(line, nearest, 0.75, 3.0);
        let previous = textureLoad(writeLine, 1u);
        let input = textureLoad(readLine, 0u);
        result[2] = vec4f(f32(textureDimensions(line)), f32(textureNumLevels(line)),
                         f32(textureDimensions(writeLine)), f32(previous.x + input.x));
        result[3] = vec4f(textureLoad(line, -1i, 0i).a, textureLoad(line, 0xffffffffu, 0u).a,
                         f32(textureDimensions(line, 1u)), f32(textureLoad(readLine, 2u).a));
        textureStore(writeLine, 1i, previous + input);
        textureStore(writeLine, -1i, vec4i(99));
        textureStore(writeLine, 0xffffffffu, vec4i(99));
    }
    """

private let lineFragmentShader = """
    @group(0) @binding(0) var line: texture_1d<f32>;
    @group(0) @binding(1) var nearest: sampler;
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return textureSample(line, nearest, 0.75); }
    """

private func explicitLineOffsetShader(_ offset: String) -> ShaderInput {
    .wgsl("""
        @group(0) @binding(0) var line: texture_1d<f32>;
        @group(0) @binding(1) var plane: texture_2d<f32>;
        @group(0) @binding(2) var volume: texture_3d<f32>;
        @group(0) @binding(3) var state: sampler;
        const texel = 1i;
        @fragment fn draw(@location(0) uv: vec2f) -> @location(0) vec4f {
            let value = textureSampleLevel(line, state, uv.x, 1.0, \(offset));
            let flat = textureSampleLevel(plane, state, uv, 0.0, vec2i(1, -2));
            let solid = textureSampleLevel(volume, state, vec3f(uv, 0.25), 0.0, vec3i(1, -2, 3));
            return value + flat + solid;
        }
        """)
}

@Test func nativeWGSLLineTextureOffsetsSurviveSPIRVInterchange() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(explicitLineOffsetShader("-1"), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let shader = try compiler.compileRenderStage(.spirv(words), entryPoint: "draw", to: .hlsl)
    guard case .text(.hlsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains(".SampleLevel("))
}

@Test(arguments: [ShaderOutputFormat.hlsl, .glsl])
func translatesNativeLineTextureSampleOffsets(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(explicitLineOffsetShader("texel - 3i"), entryPoint: "draw", to: format)
    #expect(shader.sampledTextures.map(\.dimension) == [.d1, .d2, .d3])
    guard case .text(_, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
    let sample = format == .hlsl ? ".SampleLevel(" : "textureLodOffset("
    let samples = code.split(separator: "\n").filter { $0.contains(sample) }
    try #require(samples.count == 3)
    #expect(samples[0].contains("1.0") && samples[0].hasSuffix(", (-2));"))
}

@Test(arguments: [("8i", ShaderDiagnosticCode.invalidType), ("1u", .typeMismatch), ("i32(uv.x)", .invalidType)])
func rejectsInvalidNativeLineTextureSampleOffsets(_ offset: String, _ diagnostic: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().compileRenderStage(
            explicitLineOffsetShader(offset), entryPoint: "draw", to: .hlsl)
        Issue.record("Accepted an invalid native 1D offset")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == diagnostic)
    }
}

@Test func nativeLineTextureSampleOffsetKeepsStrictWGSLAndMetalProfiles() throws {
    let input = explicitLineOffsetShader("-1")
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(input) }
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPURenderStage(input, entryPoint: "draw") }
    #expect(throws: ShaderFailure.self) { try compiler.compileRenderStage(input, entryPoint: "draw", to: .msl) }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsLineTextureLoadsSamplesAndStores(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(lineTextureShader), to: format)
    let fragment = try compiler.compileRenderStage(.wgsl(lineFragmentShader), entryPoint: "fs", to: format)
    #expect(shader.sampledTextures.first?.dimension == .d1)
    #expect(shader.storageTextures.map(\.dimension) == [.d1, .d1])
    guard case .text(_, let source) = shader.artifact, case .text(_, let draw) = fragment.artifact else {
        Issue.record("Expected source shaders"); return
    }
    switch format {
    case .wgsl:
        #expect(source.contains("texture_storage_1d<r32sint, read_write>"))
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.parseAndValidate(.wgsl(draw))
    case .msl:
        #expect(source.contains("texture1d<float, access::sample>"))
        #expect(source.contains("texture1d<int, access::read_write>"))
        #expect(draw.contains(".sample("))
    case .hlsl:
        #expect(source.contains("Texture1D<float4>") && source.contains("RWTexture1D<int>"))
        #expect(source.contains(".Load(int2(") && draw.contains(".Sample("))
    case .glsl:
        #expect(source.contains("sampler1D") && source.contains("iimage1D"))
        #expect(draw.contains("texture("))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    "let value = textureLoad(line, vec2u(0), 0u);",
    "let value = textureSampleLevel(line, nearest, vec2f(0.5), 0.0);",
    "let value = textureSampleLevel(line, nearest, 0.5, 0.0, 1i);",
    "let value = textureSampleGrad(line, nearest, 0.5, 0.0, 0.0);",
    "let value = textureGather(0, line, nearest, 0.5);",
    "textureStore(output, vec2i(0), vec4u(1));",
])
func rejectsInvalidLineTextureOperands(body: String) throws {
    let source = """
        @group(0) @binding(0) var line: texture_1d<f32>;
        @group(0) @binding(1) var nearest: sampler;
        @group(0) @binding(2) var output: texture_storage_1d<r32uint, write>;
        @compute @workgroup_size(1) fn run() { \(body) }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid 1D texture operation accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func samplesAndUpdatesLineTexturesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(lineTextureShader), to: .msl))
        let source = ShaderInput.wgsl(lineFragmentShader)
        let draw = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl), colorFormat: .rgba8Unorm)
        let line = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.textureBinding, .copyDestination], dimension: .d1))
        let view = try await device.createTextureView(texture: line, .init())
        await #expect(throws: GPUError.validation("Texture view dimension does not match the texture")) {
            try await device.createTextureView(texture: line, .init(dimension: .d2))
        }
        await #expect(throws: GPUError.self) {
            try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                usage: .textureBinding, mipLevelCount: 2, dimension: .d1))
        }
        let desc = TextureDescriptor(width: 2, height: 1, format: .r32Sint,
            usage: [.storageBinding, .copyDestination, .copySource], dimension: .d1)
        let readLine = try await device.createTexture(desc)
        let writeLine = try await device.createTexture(desc)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let sampler = try await device.createSampler(.init())
        let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ff00, 5, UInt32(bitPattern: -3)]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: line, region: .init(width: 2, height: 1)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: readLine, region: .init(width: 1, height: 1)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 12), destination: writeLine,
                                 region: .init(x: 1, width: 1, height: 1)),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 4): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): view, .init(binding: 2): readLine, .init(binding: 3): writeLine],
                             samplers: [.init(binding: 1): sampler]),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
            .copyTextureToBuffer(source: writeLine, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
            .render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 0): view], samplers: [.init(binding: 1): sampler]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        ])
        let expected: [Float] = [1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 2, 2, 0, 0, 0, 0]
        #expect(try await ticket.wait() == [expected.map(\.bitPattern), [0, 2], [0xff00ff00]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
