import SAGACompiler
import SAGAShaderIR
import Testing

@Test func nativeWGSLCooperativeMatrixRoundTripsLayoutAndMixedAccumulator() throws {
    let source = ShaderInput.wgsl("""
        enable f16;
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> aData: array<f16>;
        @group(0) @binding(1) var<storage, read> bData: array<f16>;
        @group(0) @binding(2) var<storage, read_write> cData: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoadT<coop_mat8x8<f16, A>>(&aData[0], 0u);
            let b = coopLoad<coop_mat8x8<f16, B>>(&bData[64], 12u);
            let c = coopLoadT<coop_mat8x8<f32, C>>(&cData[0], 9u);
            let result = coopMultiplyAdd(a, b, c);
            coopStoreT(result, &cData[80], 11u);
            coopStore(result, &cData[160], 7u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let generated = try compiler.compileCompute(source, to: .wgsl)
    guard case .text(.wgsl, let text) = generated.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("enable wgpu_cooperative_matrix;"))
    #expect(text.contains("coopLoadT<coop_mat8x8<f16, A>>"))
    #expect(text.contains("coopLoad<coop_mat8x8<f16, B>>"))
    #expect(text.contains("coopStoreT("))
    #expect(text.contains("0u"))
    #expect(text.contains("12u"))
    #expect(text.contains("9u"))
    #expect(text.contains("11u"))
    #expect(text.contains("7u"))
    #expect(generated.requiresCooperativeMatrix)
    #expect(generated.requiresSubgroups)
    _ = try compiler.compileCompute(.wgsl(text), to: .wgsl)
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(source) }
}
