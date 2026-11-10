import Testing
import SAGACompiler
import SAGAShaderIR

@Test func glslRenderPipelineFamilyRejectsIncompatibleTargetAndStage() throws {
    let compiler = IndependentShaderCompiler()
    let shader = ShaderInput.wgsl("@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }")
    #expect(throws: (any Error).self) {
        try compiler.compileRenderStage(shader, entryPoint: "vs", to: .msl, glslPipelineFamily: .vertex)
    }
    #expect(throws: (any Error).self) {
        try compiler.compileRenderStage(shader, entryPoint: "vs", to: .glsl, glslPipelineFamily: .mesh)
    }
    let primitive = ShaderInput.wgsl("enable wgpu_mesh_shader; @fragment fn fs(@location(0) @per_primitive value: f32) -> @location(0) vec4f { return vec4f(value); }")
    #expect(throws: ShaderFailure.self) {
        try compiler.compileRenderStage(primitive, entryPoint: "fs", to: .glsl, glslPipelineFamily: .vertex)
    }
}
