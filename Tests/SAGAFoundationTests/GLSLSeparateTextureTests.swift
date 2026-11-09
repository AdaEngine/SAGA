import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let glslSeparateTextureInput = ShaderInput.glsl("""
    #version 450
    layout(set=0,binding=0) uniform texture2D sourceImage;
    layout(set=0,binding=1) uniform sampler state;
    layout(set=0,binding=2) uniform textureCube cubeImage;
    layout(set=0,binding=3) uniform texture2DArray arrayImage;
    layout(set=0,binding=4) uniform itexture2D integerImage;
    layout(set=0,binding=5) uniform texture2DMS multisampledImage;
    layout(location=0) out vec4 outputColor;
    vec4 sampleAtLevel(texture2D image, sampler filterState, in vec2 uv) {
        return textureLod(sampler2D(image, filterState), uv, 0.0);
    }
    void main() {
        vec2 uv = vec2(0.5);
        ivec2 size = textureSize(samplerCube(cubeImage, state), 0);
        int levels = textureQueryLevels(sampler2D(sourceImage, state));
        ivec4 fetched = texelFetch(isampler2D(integerImage, state), ivec2(0), 0);
        vec4 sampleValue = texelFetch(sampler2DMS(multisampledImage, state), ivec2(0), 1);
        outputColor = texture(sampler2D(sourceImage, state), uv)
            + sampleAtLevel(sourceImage, state, uv)
            + textureGrad(sampler2D(sourceImage, state), uv, vec2(0.1), vec2(0.2))
            + textureLod(sampler2DArray(arrayImage, state), vec3(uv, 1.0), 0.0)
            + vec4(fetched) + sampleValue + vec4(float(size.x + levels));
    }
    """, stage: .fragment, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslSeparateTexturesTranslateThroughTypedSampling(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(glslSeparateTextureInput, entryPoint: "main", to: format)
    #expect(shader.sampledTextures.map(\.binding.binding) == [0, 2, 3, 4, 5])
    #expect(shader.samplerBindings.map(\.binding.binding) == [1])
    #expect(shader.sampledTextures.first?.samplingSamplers == [.init(group: 0, binding: 1)])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPURenderStage(glslSeparateTextureInput, entryPoint: "main")
    }
}

@Test(arguments: [
    "texture(samplerCube(image, state), vec3(0.0))",
    "texture(sampler2D(image, compareState), vec2(0.0))",
    "texture(image, vec2(0.0))",
    "textureGrad(sampler2D(image, state), vec2(0.0), vec3(0.0), vec2(0.0))",
    "textureLod(sampler2D(image, state), vec2(0.0), vec2(0.0))",
    "sampler2D(image, state)",
])
func glslSeparateTexturesRejectInvalidPairsAndEscapingCombinedValues(_ expression: String) {
    let source = """
        #version 450
        layout(binding=0) uniform texture2D image;
        layout(binding=1) uniform sampler state;
        layout(binding=2) uniform samplerShadow compareState;
        layout(location=0) out vec4 color;
        void main() { color = \(expression); }
        """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.glsl(source, stage: .fragment, defines: [:]))
    }
}

@Test func glslSeparateTextureVectorCastDoesNotEmitSingleConstituentSPIRVVector() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(glslSeparateTextureInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    var vectorTypes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        if opcode == 23 && count == 4 { vectorTypes.insert(words[offset + 1]) } // OpTypeVector
        if opcode == 80 && count == 4 { // OpCompositeConstruct with one constituent
            #expect(!vectorTypes.contains(words[offset + 1]))
        }
        offset += count
    }
    _ = try compiler.parseAndValidate(.spirv(words))
}

#if canImport(Metal)
@Test func glslSeparateTextureSamplingAndFetchExecuteOnMetal() async throws {
    let input = ShaderInput.glsl("""
        #version 450
        layout(local_size_x=1) in;
        layout(binding=0) uniform texture2D sourceImage;
        layout(binding=1) uniform sampler filterState;
        layout(binding=2) buffer Output { vec4 values[2]; } outputData;
        void main() {
            outputData.values[0] = textureLod(sampler2D(sourceImage, filterState), vec2(0.25, 0.5), 0.0);
            outputData.values[1] = texelFetch(sampler2D(sourceImage, filterState), ivec2(1, 0), 0);
        }
        """, stage: .compute, defines: [:])
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                          usage: [.textureBinding, .copyDestination]))
        let sampler = try await device.createSampler(.init())
        let result = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .writeTexture(destination: source, data: [255, 0, 0, 255, 0, 255, 0, 255],
                          region: .init(width: 2, height: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): result], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): source], samplers: [.init(binding: 1): sampler]),
            .copy(source: result, destination: readback, byteCount: 32),
            .read(buffer: readback, byteCount: 32),
        ]).wait()
        let expected: [Float] = [1, 0, 0, 1, 0, 1, 0, 1]
        #expect(words == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
