import SAGACompiler
import SAGAShaderIR
import Testing

@Test func wgslCooperativeMatrixEmitsEmulatedHLSL() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoadT<coop_mat8x8<f32, A>>(&data[0], 10u);
            let b = coopLoad<coop_mat8x8<f32, B>>(&data[80]);
            let c = coopLoadT<coop_mat8x8<f32, C>>(&data[144]);
            coopStoreT(coopMultiplyAdd(a, b, c), &data[0], 10u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .hlsl)
    guard case .text(.hlsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("[WaveSize(32)]"))
    #expect(source.contains("WaveReadLaneAt"))
    #expect(source.contains("WaveGetLaneIndex"))
    #expect(source.contains("asfloat("))
    #expect(source.contains("asuint("))
    #expect(shader.hlslProfile == "cs_6_6")
    #expect(shader.requiredSubgroupSize == 32)
    #expect(!shader.requiresCooperativeMatrix)
}

@Test func wgslCooperativeMatrixHLSLRejectsIncompleteWaves() {
    let incomplete = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> data: array<f32>;
        @compute @workgroup_size(16) fn main() {
            let a = coopLoad<coop_mat8x8<f32, A>>(&data[0]);
        }
        """)
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compileCompute(incomplete, to: .hlsl) }
}

@Test func wgslCooperativeMatrixHLSLGuardsOverlappingStoreRows() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let tile = coopLoad<coop_mat8x8<f32, C>>(&data[0]);
            coopStoreT(tile, &data[0], 7u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .hlsl)
    guard case .text(.hlsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("_stride = 7u;"))
    #expect(source.contains("_stride >= 8u)"))
}

@Test func wgslCooperativeMatrixHLSLMixedHalfInputsFloatAccumulator() throws {
    let input = ShaderInput.wgsl("""
        enable f16, wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> ab: array<f16>;
        @group(0) @binding(1) var<storage, read_write> accum: array<f32>;
        @compute @workgroup_size(8, 8, 1) fn main() {
            let a = coopLoad<coop_mat8x8<f16, A>>(&ab[0]);
            let b = coopLoad<coop_mat8x8<f16, B>>(&ab[0]);
            let c = coopLoad<coop_mat8x8<f32, C>>(&accum[0]);
            coopStore(coopMultiplyAdd(a, b, c), &accum[0]);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .hlsl)
    guard case .text(.hlsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("RWByteAddressBuffer"))
    #expect(source.contains(".Load<float16_t>("))
    #expect(source.contains("saga_coop_muladd("))
    #expect(source.contains("_stride >= 8u"))
    #expect(shader.requiresShaderF16)
    #expect(shader.hlslProfile == "cs_6_6")
}

@Test func wgslCooperativeMatrixHLSLRejectsHalfAccumulatorEvenWithoutStore() {
    let input = ShaderInput.wgsl("""
        enable f16, wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> data: array<f16>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoad<coop_mat8x8<f16, A>>(&data[0]);
            let b = coopLoad<coop_mat8x8<f16, B>>(&data[0]);
            let c = coopLoad<coop_mat8x8<f16, C>>(&data[0]);
            let result = coopMultiplyAdd(a, b, c);
        }
        """)
    #expect(throws: ShaderFailure.unsupported("Shader body code generation: HLSL cooperative multiply-add requires an f32 accumulator")) {
        try IndependentShaderCompiler().compileCompute(input, to: .hlsl)
    }
}

@Test func wgslCooperativeMatrixHLSLRejectsHalfAccumulatorWithStore() {
    let input = ShaderInput.wgsl("""
        enable f16, wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> data: array<f16>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoad<coop_mat16x8<f16, A>>(&data[0]);
            let b = coopLoad<coop_mat8x16<f16, B>>(&data[128]);
            let c = coopLoad<coop_mat8x8<f16, C>>(&data[256]);
            coopStore(coopMultiplyAdd(a, b, c), &data[320]);
        }
        """)
    #expect(throws: ShaderFailure.unsupported("Shader body code generation: HLSL cooperative multiply-add requires an f32 accumulator")) {
        try IndependentShaderCompiler().compileCompute(input, to: .hlsl)
    }
}

@Test func wgslCooperativeMatrixHLSLStoresHalfWithoutOverwritingNeighbors() throws {
    let input = ShaderInput.wgsl("""
        enable f16, wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> data: array<f16>;
        @compute @workgroup_size(32) fn main() {
            let tile = coopLoad<coop_mat8x8<f16, C>>(&data[1]);
            coopStoreT(tile, &data[1], 9u);
            coopStore(tile, &data[3], 10u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .hlsl)
    guard case .text(.hlsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.split(separator: ".Store<float16_t>(").count == 5)
    #expect(source.contains("_stride >= 8u"))
    #expect(source.contains("* 2u"))
    #expect(!source.contains("Interlocked"))
    #expect(shader.requiresShaderF16)
}

@Test func wgslCooperativeMatrixHLSLRectangularTiles() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoadT<coop_mat16x8<f32, A>>(&data[0], 18u);
            let b = coopLoad<coop_mat8x16<f32, B>>(&data[256], 18u);
            let c = coopLoadT<coop_mat8x8<f32, C>>(&data[512], 10u);
            coopStoreT(coopMultiplyAdd(a, b, c), &data[512], 10u);
            let d = coopLoad<coop_mat16x16<f32, C>>(&data[768], 17u);
            coopStore(d, &data[768], 17u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .hlsl)
    guard case .text(.hlsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("saga_coop_muladd("))
    #expect(source.contains("16u"))
    #expect(source.contains("WaveReadLaneAt"))
}
