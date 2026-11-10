import Testing
import SAGACompiler
import SAGAFrontends
import SAGAShaderIR

// Independently authored regression cases for Naga's abstract integer concretization preferences.
@Suite struct WGSLConcretizationPreferenceTests {
    @Test func nagaProfileConcretizesAbstractIntegersUsingFirstRepresentableType() throws {
        for declarations in [
            "let x = 2147483648; let y: u32 = x;",
            "let x = 4294967296; let y: f32 = x;",
            "let x = vec2(1, 2147483648); let y: vec2<u32> = x;",
            "let x = array(1, 4294967296); let y: array<f32, 2> = x;"
        ] {
            let source = ShaderInput.wgsl("@compute @workgroup_size(1) fn main() { \(declarations) }")
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(source) }
            #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compileCompute(source, to: .wgsl) }
            let naga = IndependentShaderCompiler(wgslSourceProfile: .nagaNative)
            _ = try naga.compileCompute(source, to: .wgsl)
            #expect(throws: ShaderFailure.self) { try naga.compileWebGPUCompute(source) }
        }
    }
}
