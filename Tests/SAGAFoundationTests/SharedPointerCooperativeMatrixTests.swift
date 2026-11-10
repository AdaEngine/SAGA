import SAGACompiler
import SAGAShaderIR
import Testing

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func sharedPointerSpecializationPreservesCooperativeMatrixOperands(_ format: ShaderOutputFormat) throws {
    let source = """
        enable wgpu_cooperative_matrix;
        requires unrestricted_pointer_parameters;
        @group(0) @binding(0) var<storage, read_write> data: array<f32, 256>;
        fn transform(p: ptr<storage, f32, read_write>) {
            let a = coopLoad<coop_mat8x8<f32, A>>(p, 8u);
            let b = coopLoad<coop_mat8x8<f32, B>>(p, 8u);
            let c = coopLoad<coop_mat8x8<f32, C>>(p, 8u);
            coopStore(coopMultiplyAdd(a, b, c), p, 8u);
        }
        @compute @workgroup_size(32) fn main() { transform(&data[0]); }
        """
    let artifact = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: format)
    guard case .text(let actualFormat, let output) = artifact.artifact, actualFormat == format else {
        throw ShaderFailure.incompatibleArtifact
    }
    switch format {
    case .msl:
        #expect(output.contains("simdgroup_load("))
        #expect(output.contains("simdgroup_multiply_accumulate("))
        #expect(output.contains("simdgroup_store("))
    case .hlsl:
        #expect(output.contains("saga_coop_muladd("))
        #expect(output.contains("WaveReadLaneAt"))
        #expect(output.contains("asfloat("))
        #expect(output.contains("asuint("))
    case .glsl:
        #expect(output.contains("coopMatLoad"))
        #expect(output.contains("coopMatMulAdd"))
        #expect(output.contains("coopMatStore"))
    default:
        Issue.record("Unexpected output format")
    }
}
