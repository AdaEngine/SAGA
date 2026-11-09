import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let glslShadowInput = ShaderInput.glsl("""
    #version 450
    layout(binding=0) uniform texture2D nearImage;
    layout(binding=1) uniform texture2D farImage;
    layout(binding=2) uniform samplerShadow compareState;
    layout(location=0) out vec4 color;
    float sampleDepth(texture2D image, samplerShadow state, float reference) {
        return texture(sampler2DShadow(image, state), vec3(0.5, 0.5, reference));
    }
    float throughHelper(texture2D image, samplerShadow state, float reference) {
        return sampleDepth(image, state, reference);
    }
    void main() {
        color = vec4(throughHelper(nearImage, compareState, 0.5),
                     throughHelper(farImage, compareState, 0.5), 0.0, 1.0);
    }
    """, stage: .fragment, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslShadowTextureInferenceFollowsNestedParameters(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let stage = try compiler.compileRenderStage(glslShadowInput, entryPoint: "main", to: format)
    #expect(stage.sampledTextures.map(\.isDepth) == [true, true])
    #expect(stage.sampledTextures.map(\.binding.binding) == [0, 1])
    #expect(stage.samplerBindings.map(\.binding.binding) == [2])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = stage.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains("textureSampleCompare("))
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPURenderStage(glslShadowInput, entryPoint: "main")
    }
}

@Test func glslShadowTextureInferenceRejectsMixedOrdinaryAndComparisonUses() throws {
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform texture2D image;
        layout(binding=1) uniform samplerShadow compareState;
        layout(binding=2) uniform sampler regularState;
        layout(location=0) out vec4 color;
        float compareAt(texture2D value, samplerShadow state) {
            return texture(sampler2DShadow(value, state), vec3(0.5));
        }
        void main() {
            color = texture(sampler2D(image, regularState), vec2(0.5)) + vec4(compareAt(image, compareState));
        }
        """, stage: .fragment, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Accepted incompatible uses of one texture")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .unsupportedConstruct)
        #expect(diagnostics.first?.message.contains("ordinary and comparison") == true)
    }
}

@Test func glslShadowTextureInferenceDoesNotPromoteUnrelatedQueryBindings() throws {
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform texture2D image;
        layout(binding=1) uniform texture2D unrelated;
        layout(binding=2) uniform samplerShadow compareState;
        layout(binding=3) uniform sampler regularState;
        layout(location=0) out vec4 color;
        float widthOf(texture2D value, sampler state) {
            return float(textureSize(sampler2D(value, state), 0).x);
        }
        void main() {
            color = vec4(texture(sampler2DShadow(image, compareState), vec3(0.5)),
                         widthOf(image, regularState), widthOf(unrelated, regularState), 1.0);
        }
        """, stage: .fragment, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Inferred depth for an unrelated texture binding")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .unsupportedConstruct)
        #expect(diagnostics.first?.message.contains("parameter cannot receive both") == true)
    }
}

#if canImport(Metal)
@Test func glslShadowHelpersCompareBothBoundDepthTexturesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertexSource = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0, 1); }
            if index == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn clearColor() -> @location(0) vec4f { return vec4f(0); }
        """)
    let vertex = try compiler.compileRenderStage(vertexSource, entryPoint: "vs", to: .msl)
    let clearColor = try compiler.compileRenderStage(vertexSource, entryPoint: "clearColor", to: .msl)
    let fragment = try compiler.compileRenderStage(glslShadowInput, entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let clear = try await device.createRenderPipeline(vertex: vertex, fragment: clearColor, colorFormat: .rgba8Unorm,
                                                          depthState: .init(compare: .always))
        let sample = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let descriptor = TextureDescriptor(width: 1, height: 1, format: .depth32Float,
                                           usage: [.renderAttachment, .textureBinding])
        let near = try await device.createTexture(descriptor)
        let far = try await device.createTexture(descriptor)
        let sampler = try await device.createSampler(.init(compare: .less))
        let readback = try await device.createBuffer(.init(size: 256, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .render(pipeline: clear, target: target, clear: .transparentBlack, vertexCount: 0,
                    depth: .init(texture: near, loadAction: .clear(0.25))),
            .render(pipeline: clear, target: target, clear: .transparentBlack, vertexCount: 0,
                    depth: .init(texture: far, loadAction: .clear(0.75))),
            .render(pipeline: sample, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 0): near, .init(binding: 1): far],
                    samplers: [.init(binding: 2): sampler]),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(words == [[0xff00ff00]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
