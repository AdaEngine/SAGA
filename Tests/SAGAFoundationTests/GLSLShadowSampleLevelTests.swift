import SAGACompiler
import SAGAShaderIR
import Testing

private let shadowSampleLevelSource = """
    #version 450
    layout(binding=0) uniform texture2D depthImage;
    layout(binding=1) uniform texture2DArray depthArray;
    layout(binding=2) uniform textureCube depthCube;
    layout(binding=3) uniform samplerShadow state;
    layout(location=0) in vec4 coord;
    layout(location=0) out vec4 color;
    float arrayGrad() {
        return textureGradOffset(sampler2DArrayShadow(depthArray, state), coord, vec2(0.2), vec2(0.4), ivec2(1, -2));
    }
    void main() {
        float a = textureProjGradOffset(sampler2DShadow(depthImage, state), coord, vec2(0.1), vec2(0.3), ivec2(-1, 2));
        float b = textureGrad(samplerCubeShadow(depthCube, state), coord, vec3(0.2), vec3(0.4));
        float c = textureLodOffset(sampler2DShadow(depthImage, state), coord.xyz, coord.w, ivec2(1));
        color = vec4(a, b, c, arrayGrad());
    }
    """

@Test func glslShadowSampleLevelsPreserveComparisonThroughSPIRV() throws {
    let compiler = IndependentShaderCompiler()
    let glsl = ShaderInput.glsl(shadowSampleLevelSource, stage: .fragment, defines: [:])
    guard case .spirv(let words) = try compiler.compile(glsl, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    for input in [glsl, .spirv(words)] {
        let metal = try compiler.compileRenderStage(input, entryPoint: "main", to: .msl)
        guard case .text(.msl, let msl) = metal.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(msl.split(separator: ".sample_compare(").count == 5)
        #expect(msl.split(separator: "gradient2d(").count == 3)
        #expect(msl.contains("gradientcube("))
        #expect(msl.contains("level("))
        #expect(msl.contains(" / "))
        let directX = try compiler.compileRenderStage(input, entryPoint: "main", to: .hlsl)
        guard case .text(.hlsl, let hlsl) = directX.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(directX.hlslProfile == "ps_6_8")
        #expect(hlsl.split(separator: ".SampleCmpGrad(").count == 4)
        #expect(hlsl.contains(".SampleCmpLevel("))
        let openGL = try compiler.compileRenderStage(input, entryPoint: "main", to: .glsl)
        guard case .text(.glsl, let source) = openGL.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.split(separator: "textureGradOffset(").count == 3)
        #expect(source.contains("textureGrad("))
        #expect(source.contains("textureLodOffset("))
        #expect(throws: ShaderFailure.self) { try compiler.compileWebGPURenderStage(input, entryPoint: "main") }
    }
}

@Test(arguments: ["vec3(0.2), vec2(0.4), ivec2(0)", "vec2(0.2), vec2(0.4), ivec2(8)"])
func glslShadowSampleGradientRejectsInvalidOperands(_ operands: String) {
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform texture2D depthImage;
        layout(binding=1) uniform samplerShadow state;
        layout(location=0) out float color;
        void main() {
            color = textureGradOffset(sampler2DShadow(depthImage, state), vec3(0.5), \(operands));
        }
        """, stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}

@Test(arguments: [("0.0", "ps_6_0", "SampleCmpLevelZero"), ("1.5", "ps_6_7", "SampleCmpLevel")])
func glslShadowSampleLODSelectsHLSLProfile(_ lod: String, _ profile: String, _ method: String) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform texture2D depthImage;
        layout(binding=1) uniform samplerShadow state;
        layout(location=0) out float color;
        void main() {
            color = textureLod(sampler2DShadow(depthImage, state), vec3(0.5), \(lod));
        }
        """, stage: .fragment, defines: [:])
    let hlsl = try compiler.compileRenderStage(input, entryPoint: "main", to: .hlsl)
    guard case .text(.hlsl, let source) = hlsl.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(hlsl.hlslProfile == profile)
    #expect(source.contains(".\(method)("))
    if lod == "0.0" {
        let wgsl = try compiler.compileWebGPURenderStage(input, entryPoint: "main")
        guard case .text(.wgsl, let source) = wgsl.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains("textureSampleCompareLevel("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    } else {
        #expect(throws: ShaderFailure.self) { try compiler.compileWebGPURenderStage(input, entryPoint: "main") }
    }
}
