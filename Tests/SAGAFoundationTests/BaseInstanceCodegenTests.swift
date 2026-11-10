import Testing
import SAGACompiler
import SAGAShaderIR

@Test func glslBaseInstanceEmitsVertexInputsAcrossNativeTargets() throws {
    let input = ShaderInput.glsl("""
        #version 460 core
        void main() {
            gl_Position = vec4(float(gl_BaseInstance), float(gl_InstanceID), 0.0, 1.0);
        }
        """, stage: .vertex, defines: [:])
    let compiler = IndependentShaderCompiler()
    for format in [ShaderOutputFormat.msl, .hlsl, .glsl] {
        let result = try compiler.compileRenderStage(input, entryPoint: "main", to: format)
        guard case .text(_, let code) = result.artifact else { throw ShaderFailure.incompatibleArtifact }
        switch format {
        case .msl:
            #expect(code.contains("[[base_instance]]"))
            #expect(code.contains("[[instance_id]]"))
        case .hlsl:
            #expect(result.hlslProfile == "vs_6_8")
            #expect(code.indices.filter { code[$0...].hasPrefix("SV_StartInstanceLocation") }.count == 1)
            #expect(code.contains("SV_InstanceID"))
        case .glsl:
            #expect(code.contains("#version 460 core"))
            #expect(code.contains("uint(gl_BaseInstance)"))
        default: Issue.record("Unexpected target")
        }
    }
}

@Test(arguments: ["float(gl_BaseInstance)", "float(gl_InstanceID) + float(gl_BaseInstance)"])
func hlslBaseInstanceUsesOneSystemParameter(expression: String) throws {
    let input = ShaderInput.glsl("""
        #version 460 core
        void main() { gl_Position = vec4(\(expression), 0.0, 0.0, 1.0); }
        """, stage: .vertex, defines: [:])
    let result = try IndependentShaderCompiler().compileRenderStage(input, entryPoint: "main", to: .hlsl)
    guard case .text(.hlsl, let code) = result.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(result.hlslProfile == "vs_6_8")
    #expect(code.indices.filter { code[$0...].hasPrefix("SV_StartInstanceLocation") }.count == 1)
}
