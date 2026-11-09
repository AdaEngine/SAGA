import SAGACompiler
import SAGAShaderIR
import Testing

private let lineGradSource = ShaderInput.glsl("""
    #version 450
    layout(binding=0) uniform texture1D image;
    layout(binding=1) uniform sampler state;
    layout(location=0) out vec4 color;
    void main() {
        color = textureGradOffset(sampler1D(image, state), 0.5, 0.25, 0.5, 1);
    }
    """, stage: .fragment, defines: [:])

@Test func glslLineGradientsRoundTripThroughNativeTargets() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(lineGradSource, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    for input in [lineGradSource, .spirv(words)] {
        for (format, spelling): (ShaderOutputFormat, String) in [(.glsl, "textureGradOffset("), (.hlsl, ".SampleGrad(")] {
            let shader = try compiler.compileRenderStage(input, entryPoint: "main", to: format)
            guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
            #expect(text.contains(spelling))
        }
        for format: ShaderOutputFormat in [.wgsl, .msl] {
            #expect(throws: ShaderFailure.self) { try compiler.compileRenderStage(input, entryPoint: "main", to: format) }
        }
    }
}

@Test(arguments: ["ivec2(1, -2)", "offset", "ivec2(offset.x + 1, -1)"])
func glslTextureOffsetsFoldTypedConstantExpressions(_ expression: String) throws {
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform texture2D image;
        layout(binding=1) uniform sampler state;
        layout(location=0) out vec4 color;
        const ivec2 offset = ivec2(1, -2);
        void main() { color = textureLodOffset(sampler2D(image, state), vec2(0.5), 0.0, \(expression)); }
        """, stage: .fragment, defines: [:])
    let compiler = IndependentShaderCompiler()
    let stage = try compiler.compileWebGPURenderStage(input, entryPoint: "main")
    guard case .text(.wgsl, let text) = stage.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("textureSampleLevel("))
    _ = try compiler.parseAndValidate(.wgsl(text))
}

@Test(arguments: ["ivec2(8)", "ivec3(0)", "ivec2(gl_FragCoord.xy)"])
func glslTextureOffsetsRejectInvalidRangesAndRuntimeValues(_ expression: String) {
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform texture2D image;
        layout(binding=1) uniform sampler state;
        layout(location=0) out vec4 color;
        void main() { color = textureOffset(sampler2D(image, state), vec2(0.5), \(expression)); }
        """, stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}
