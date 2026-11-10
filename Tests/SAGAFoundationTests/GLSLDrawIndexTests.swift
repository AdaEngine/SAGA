import Testing
import SAGACompiler
import SAGAShaderIR

@Test func glslDrawIndexUsesCoreDrawIDForWGSLAndSPIRV() throws {
    let input = ShaderInput.wgsl("""
        enable draw_index;
        struct Inputs { @builtin(draw_index) draw: u32, @builtin(vertex_index) vertex: u32 }
        @vertex fn vs(input: Inputs) -> @builtin(position) vec4f {
            return vec4f(f32(input.draw), f32(input.vertex), 0, 1);
        }
        """)
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    for shader in [input, .spirv(words)] {
        let result = try compiler.compileRenderStage(shader, entryPoint: "vs", to: .glsl)
        guard case .text(.glsl, let code) = result.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(code.contains("#version 460 core"))
        #expect(code.contains("uint(gl_DrawID)"))
        #expect(code.contains("uint(gl_VertexID)"))
    }
}
