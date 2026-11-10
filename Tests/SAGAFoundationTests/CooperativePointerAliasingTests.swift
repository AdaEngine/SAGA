import SAGACompiler
import SAGAShaderIR
import Testing

private func cooperativePointerShader(body: String, arguments: String) -> ShaderInput {
    .wgsl("""
        enable wgpu_cooperative_matrix;
        requires unrestricted_pointer_parameters;
        @group(0) @binding(0) var<storage, read_write> first: array<f32, 256>;
        @group(0) @binding(1) var<storage, read_write> second: array<f32, 256>;
        fn usePointers(a: ptr<storage, f32, read_write>, b: ptr<storage, f32, read_write>) {
            \(body)
        }
        @compute @workgroup_size(32) fn main() { usePointers(\(arguments)); }
        """)
}

@Test(arguments: [
    "let m = coopLoad<coop_mat8x8<f32, C>>(b, 8u); coopStore(m, a, 8u);",
    "let m = coopLoad<coop_mat8x8<f32, C>>(a, 8u); coopStore(m, b, 8u);",
])
func rejectsConflictingCooperativePointerAliases(body: String) throws {
    let source = cooperativePointerShader(body: body, arguments: "&first[0], &first[0]")
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source, capabilities: .referenceDefault.union(.cooperativeMatrix))
        Issue.record("Accepted conflicting cooperative pointer aliases")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .invalidReference && $0.message.contains("Aliased pointer arguments") })
    }
}

@Test func rejectsCooperativeReadAliasingGlobalWrite() throws {
    let source = cooperativePointerShader(
        body: "let m = coopLoad<coop_mat8x8<f32, C>>(a, 8u); first[0] = 1.0;",
        arguments: "&first[0], &second[0]"
    )
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source, capabilities: .referenceDefault.union(.cooperativeMatrix))
        Issue.record("Accepted cooperative pointer aliasing a written global")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .invalidReference && $0.message.contains("aliases a global") })
    }
}

@Test func acceptsReadOnlyCooperativePointerAliases() throws {
    let source = cooperativePointerShader(
        body: "let x = coopLoad<coop_mat8x8<f32, C>>(a, 8u); let y = coopLoad<coop_mat8x8<f32, C>>(b, 8u);",
        arguments: "&first[0], &first[0]"
    )
    _ = try IndependentShaderCompiler().parseAndValidate(source, capabilities: .referenceDefault.union(.cooperativeMatrix))
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func compilesDistinctCooperativePointerArguments(format: ShaderOutputFormat) throws {
    let source = cooperativePointerShader(
        body: "let m = coopLoad<coop_mat8x8<f32, C>>(a, 8u); coopStore(m, b, 8u);",
        arguments: "&first[0], &second[0]"
    )
    _ = try IndependentShaderCompiler().compileCompute(source, to: format)
}
