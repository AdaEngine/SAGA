import SAGACompiler
import SAGAShaderIR
import Testing

@Test func wgslCooperativeMatrixGLSLInitializesPrivateTiles() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        var<private> left: coop_mat8x8<f32, A>;
        var<private> right: coop_mat8x8<f32, B>;
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let accumulator = coopLoad<coop_mat8x8<f32, C>>(&data[0]);
            coopStore(coopMultiplyAdd(left, right, accumulator), &data[0]);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .glsl)
    guard case .text(.glsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("gl_MatrixUseA>(0.0)"))
    #expect(source.contains("gl_MatrixUseB>(0.0)"))
}

@Test func wgslCooperativeMatrixEmitsGLSLKHRCooperativeMatrix() throws {
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
    let artifact = try IndependentShaderCompiler().compileCompute(input, to: .glsl)
    guard case .text(.glsl, let source) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("#extension GL_KHR_cooperative_matrix : require"))
    #expect(source.contains("#extension GL_KHR_memory_scope_semantics : require"))
    #expect(source.contains("coopMatLoad"))
    #expect(source.contains("coopMatMulAdd"))
    #expect(source.contains("coopMatStore"))
    #expect(source.contains("gl_CooperativeMatrixLayoutRowMajor"))
    #expect(source.contains("/ 7u"))
}

@Test func wgslCooperativeMatrixGLSLZeroInitializesMixedHalfTiles() throws {
    let input = ShaderInput.wgsl("""
        enable f16;
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> factors: array<f16>;
        @group(0) @binding(1) var<storage, read_write> accumulators: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoadT<coop_mat8x8<f16, A>>(&factors[0], 10u);
            let b = coopLoad<coop_mat8x8<f16, B>>(&factors[80]);
            let c = coopLoadT<coop_mat8x8<f32, C>>(&accumulators[0], 10u);
            coopStoreT(coopMultiplyAdd(a, b, c), &accumulators[0], 10u);
        }
        """)
    let artifact = try IndependentShaderCompiler().compileCompute(input, to: .glsl)
    guard case .text(.glsl, let source) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseA>"))
    #expect(source.contains("coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator>"))
    #expect(source.contains("float16_t(0.0)"))
    #expect(source.contains("coopMatMulAdd("))
}

@Test func rectangularCooperativeMatricesUseLayoutSpecificBounds() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoadT<coop_mat8x16<f32, A>>(&data[0], 8i);
            let b = coopLoad<coop_mat16x8<f32, B>>(&data[0], 16i);
            let transposed = coopLoad<coop_mat8x16<f32, A>>(&data[0], -1i);
            let repeated = coopLoadT<coop_mat16x8<f32, B>>(&data[0], 0i);
            let c = coopLoadT<coop_mat16x16<f32, C>>(&data[0], 16i);
            coopStoreT(coopMultiplyAdd(a, b, c), &data[0], 16i);
        }
        """)
    let compiler = IndependentShaderCompiler()
    for format in [ShaderOutputFormat.hlsl, .msl] {
        let shader = try compiler.compileCompute(input, to: format)
        #expect(shader.requiredSubgroupSize == 32)
        guard case .text(_, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains(format == .hlsl ? "saga_coop_muladd(" : "simdgroup_multiply_accumulate("))
    }
    for format in [ShaderOutputFormat.glsl, .wgsl] {
        let shader = try compiler.compileCompute(input, to: format)
        guard case .text(_, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains("/ 15u"))
        #expect(source.contains("/ 7u"))
        #expect(source.contains(format == .glsl ? "int(0u - uint(1)) >= 0" : "(-1i) >= 0i"))
        #expect(source.contains(">= 16u"))
        #expect(source.contains(">= 0"))
        if format == .glsl {
            #expect(source.contains("gl_ScopeSubgroup, 16, 8, gl_MatrixUseA"))
            #expect(source.contains("gl_ScopeSubgroup, 8, 16, gl_MatrixUseB"))
        } else {
            #expect(source.contains("coop_mat8x16<f32, A>"))
            _ = try compiler.compileCompute(.wgsl(source), to: .wgsl)
        }
    }
}
