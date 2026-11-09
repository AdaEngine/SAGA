import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslTypedLocalInitializerListsPreserveElementOrder() throws {
    let input = ShaderInput.glsl("""
        #version 440 core
        layout(local_size_x = 1) in;
        layout(std430, binding = 0) buffer Result { float value; } data;
        struct Payload { vec2 pair; float samples[2]; };
        int first() { return 1; }
        float second() { return 2.0; }
        void main() {
            const vec2 bias = { -1.0, 2, };
            Payload payload = { { first(), second() }, { data.value, bias.y }, };
            mat2 matrix = { { 1, 0 }, { 0, 1 } };
            data.value = (matrix * payload.pair).x + payload.samples[0];
        }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let calls = main.statements.values.compactMap { statement -> String? in
        guard case .call(let function, _, _) = statement.kind else { return nil }
        return module.functions.values[function.index].name
    }
    #expect(calls == ["first", "second"])
    let payload = try #require(main.expressions.values.first {
        guard case .compose(let type, _) = $0.kind,
              case .structure = module.types.values[type.index].kind else { return false }
        return true
    })
    guard case .compose(_, let members) = payload.kind,
          case .compose(_, let components) = main.expressions.values[members[0].index].kind,
          case .cast(let first, .f32) = main.expressions.values[components[0].index].kind else {
        Issue.record("Expected the first vector lane to convert the first call's result to float")
        return
    }
    #expect(main.statements.values.contains {
        if case .call(let function, _, let result) = $0.kind {
            return result == first && module.functions.values[function.index].name == "first"
        }
        return false
    })
    for format in [ShaderOutputFormat.wgsl, .msl] {
        _ = try compiler.compileCompute(input, to: format)
    }
    let spirv = try compiler.compile(input, to: .spirv)
    guard case .spirv(let words) = spirv else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.spirv(words))
}

@Test(arguments: [
    "vec2 value = { 1 };",
    "vec2 value = { 1, 2, 3 };",
    "float value[2] = { true, 2 };",
    "Payload value = { { 1, 2 }, { 3, 4, 5 } };",
    "mat2 value = { { 1 }, { 2, 3 } };",
])
func glslTypedLocalInitializerListsRejectWrongElementTypesOrCounts(_ declaration: String) {
    let input = ShaderInput.glsl("""
        #version 440 core
        layout(local_size_x = 1) in;
        struct Payload { vec2 pair; float samples[2]; };
        void main() { \(declaration) }
        """, stage: .compute, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Expected a type mismatch")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.map(\.code) == [.typeMismatch])
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}
