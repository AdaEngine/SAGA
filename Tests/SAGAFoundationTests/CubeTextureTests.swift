import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let cubeSamplingShader = """
    @group(0) @binding(0) var cube: texture_cube<f32>;
    @group(0) @binding(1) var cubes: texture_cube_array<f32>;
    @group(0) @binding(2) var depth: texture_depth_cube;
    @group(0) @binding(3) var depths: texture_depth_cube_array;
    @group(0) @binding(4) var nearest: sampler;
    @group(0) @binding(5) var comparison: sampler_comparison;
    @group(0) @binding(6) var<storage, read_write> out: array<vec4f, 11>;
    @compute @workgroup_size(1) fn run() {
        let direction = vec3f(1, 0, 0);
        out[0] = textureSampleLevel(cube, nearest, direction, 0.0);
        out[1] = textureSampleLevel(cube, nearest, vec3f(-1, 0, 0), 0.0);
        out[2] = textureSampleLevel(cube, nearest, vec3f(0, 1, 0), 0.0);
        out[3] = textureSampleLevel(cube, nearest, vec3f(0, -1, 0), 0.0);
        out[4] = textureSampleLevel(cube, nearest, vec3f(0, 0, 1), 0.0);
        out[5] = textureSampleLevel(cube, nearest, vec3f(0, 0, -1), 0.0);
        out[6] = textureSampleGrad(cubes, nearest, direction, -5i, vec3f(0), vec3f(0));
        out[7] = textureSampleLevel(cubes, nearest, direction, 0xffffffffu, 0.0);
        out[8] = textureGather(2, cubes, nearest, direction, 1u);
        let size = textureDimensions(cubes, 0u);
        out[9] = vec4f(f32(size.x), f32(size.y), f32(textureNumLayers(cubes)), f32(textureNumLevels(cube)));
        out[10] = vec4f(textureSampleLevel(depth, nearest, direction, 0u),
            textureSampleCompareLevel(depth, comparison, direction, 0.5),
            textureSampleCompareLevel(depths, comparison, direction, -1i, 0.5),
            textureGatherCompare(depths, comparison, direction, 0xffffffffu, 0.5).x);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsCubeTextureSamplingAndQueries(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(cubeSamplingShader), to: format)
    #expect(shader.sampledTextures.map(\.isArrayed) == [false, true, false, true])
    #expect(shader.sampledTextures.map(\.isDepth) == [false, false, true, true])
    #expect(shader.sampledTextures.map(\.dimension) == [.cube, .cube, .cube, .cube])
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("texture_depth_cube_array"))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl:
        #expect(body.contains("texturecube_array<float, access::sample>"))
        #expect(body.contains("depthcube<float, access::sample>"))
        #expect(body.contains("gradientcube(") && body.contains("component::z"))
    case .hlsl:
        #expect(body.contains("TextureCubeArray<float4>") && body.contains("TextureCube<float>"))
        #expect(body.contains(".SampleCmpLevelZero(") && body.contains(".GatherBlue("))
    case .glsl:
        #expect(body.contains("uniform samplerCubeArrayShadow"))
        #expect(body.contains("#extension GL_EXT_texture_shadow_lod : require"))
        #expect(body.contains("textureGather(") && body.contains("textureLod("))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    "textureLoad(cube, vec2i(0), 0i)",
    "textureSampleLevel(cube, nearest, vec2f(1), 0.0)",
    "textureSampleLevel(cube, nearest, vec3f(1), 0.0, vec2i(0))",
    "textureSampleGrad(cube, nearest, vec3f(1), vec2f(0), vec2f(0))",
    "textureNumLayers(cube)",
    "textureSampleBaseClampToEdge(cube, nearest, vec3f(1))",
])
func rejectsInvalidCubeTextureOperations(expression: String) throws {
    let source = """
        @group(0) @binding(0) var cube: texture_cube<f32>;
        @group(0) @binding(1) var nearest: sampler;
        @compute @workgroup_size(1) fn run() { let value = \(expression); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid cube texture operation accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

private let cubeDepthWriter = """
    @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0); }
    """

private let cubeFragmentShader = """
    @group(0) @binding(0) var cube: texture_cube<f32>;
    @group(0) @binding(1) var cubes: texture_cube_array<f32>;
    @group(0) @binding(2) var depths: texture_depth_cube_array;
    @group(0) @binding(3) var nearest: sampler;
    @group(0) @binding(4) var comparison: sampler_comparison;
    @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        if index == 0u { return vec4f(-1, -1, 0, 1); }
        if index == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> @location(0) vec4f {
        let direction = vec3f(1, 0, 0);
        let a = textureSample(cube, nearest, direction);
        let b = textureSampleBias(cubes, nearest, direction, 1i, 0.25);
        let cmp = textureSampleCompare(depths, comparison, direction, 1i, 0.5);
        let raw = textureSample(depths, nearest, direction, 1i);
        return vec4f(a.b, b.b, cmp, raw);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsImplicitCubeSamplingAndComparison(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(cubeFragmentShader), entryPoint: "fs", to: format)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl: _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl: #expect(body.contains("bias(") && body.contains(".sample_compare("))
    case .hlsl: #expect(body.contains(".SampleBias(") && body.contains(".SampleCmp("))
    case .glsl:
        #expect(body.contains("samplerCubeArrayShadow") && body.contains("texture("))
        #expect(!body.contains("GL_EXT_texture_shadow_lod"))
    default: Issue.record("Unexpected format")
    }
}

#if canImport(Metal)
@Test func samplesCubeFacesAndViewRelativeCubeIndicesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let renderSource = ShaderInput.wgsl(cubeDepthWriter)
        let render = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(renderSource, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(renderSource, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm, depthState: .init())
        let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(cubeSamplingShader), to: .msl))
        let drawSource = ShaderInput.wgsl(cubeFragmentShader)
        let draw = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(drawSource, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(drawSource, entryPoint: "fs", to: .msl), colorFormat: .rgba8Unorm)
        let color = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
            usage: [.copyDestination, .textureBinding], mipLevelCount: 2, arrayLayerCount: 13))
        let depth = try await device.createTexture(.init(width: 4, height: 4, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding], mipLevelCount: 2, arrayLayerCount: 13))
        let cube = TextureViewDescriptor(baseMipLevel: 1, dimension: .cube, baseArrayLayer: 7)
        let array = TextureViewDescriptor(baseMipLevel: 1, dimension: .cubeArray, baseArrayLayer: 1)
        let singleColor = try await device.createTextureView(texture: color, cube)
        let arrayColor = try await device.createTextureView(texture: color, array)
        let singleDepth = try await device.createTextureView(texture: depth, cube)
        let arrayDepth = try await device.createTextureView(texture: depth, array)
        let firstFace = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 1))
        let lastFace = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 7))
        await #expect(throws: GPUError.self) {
            try await device.createTextureView(texture: color, .init(dimension: .cubeArray))
        }
        await #expect(throws: GPUError.self) {
            try await device.createTextureView(texture: color, .init(dimension: .cube, arrayLayerCount: 5))
        }
        let rectangular = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: .textureBinding, arrayLayerCount: 6))
        await #expect(throws: GPUError.self) {
            try await device.createTextureView(texture: rectangular, .init(dimension: .cube))
        }
        let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let nearest = try await device.createSampler(.init())
        let compare = try await device.createSampler(.init(compare: .less))
        let output = try await device.createBuffer(.init(size: 176, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
        let upload = try await device.createBuffer(.init(size: 7 * 512, usage: [.copySource, .copyDestination]))
        let faces: [(UInt32, UInt32)] = [(1, 0xff0000ff), (7, 0xffff0000), (8, 0xff0000ff), (9, 0xff00ff00),
                                      (10, 0xff00ffff), (11, 0xffffff00), (12, 0xffff00ff)]
        var commands: [BufferCommand] = []
        for (index, (layer, texel)) in faces.enumerated() {
            let offset = UInt64(index) * 512
            commands += [.write(buffer: upload, offset: offset, words: [texel, texel]),
                         .write(buffer: upload, offset: offset + 256, words: [texel, texel]),
                         .copyBufferToTexture(source: upload, layout: .init(offset: offset, bytesPerRow: 256),
                                              destination: color, region: .init(width: 2, height: 2, mipLevel: 1, arrayLayer: layer))]
        }
        func dispatch(_ cube: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 6): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): cube, .init(binding: 1): arrayColor,
                                        .init(binding: 2): singleDepth, .init(binding: 3): arrayDepth],
                             samplers: [.init(binding: 4): nearest, .init(binding: 5): compare])
        }
        await #expect(throws: GPUError.validation("Invalid sampled texture binding")) {
            try await device.submitCommands([dispatch(arrayColor)])
        }
        commands += [
            .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                    depth: .init(texture: firstFace, loadAction: .clear(0.25))),
            .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                    depth: .init(texture: lastFace, loadAction: .clear(0.75))),
            dispatch(singleColor), .copy(source: output, destination: read, byteCount: 176), .read(buffer: read, byteCount: 176),
            .render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 0): singleColor, .init(binding: 1): arrayColor, .init(binding: 2): arrayDepth],
                    samplers: [.init(binding: 3): nearest, .init(binding: 4): compare]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
        ]
        let expected: [Float] = [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1,
                                 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 0.75, 1, 0, 1]
        let ticket = try await device.submitCommands(commands)
        #expect(try await ticket.wait() == [expected.map(\.bitPattern), [0xbfffffff, 0xbfffffff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
