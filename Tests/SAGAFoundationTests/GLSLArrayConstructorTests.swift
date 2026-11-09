import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslRuntimeFixedArrayConstructors() throws {
    let input = ShaderInput.glsl("""
        #version 440 core
        layout(local_size_x = 1) in;
        layout(std430, binding = 0) buffer Result { float value; } data;
        void main() {
            float values[2] = float[2](data.value, 2);
            data.value = float[1](values[0] + values[1])[0];
        }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let arrayCompositions = main.expressions.values.compactMap { expression -> Int? in
        guard case .compose(let type, let components) = expression.kind,
              case .array = module.types.values[type.index].kind else { return nil }
        return components.count
    }
    #expect(arrayCompositions == [2, 1])
    for format in [ShaderOutputFormat.wgsl, .msl] {
        _ = try compiler.compileCompute(input, to: format)
    }
    let spirv = try compiler.compile(input, to: .spirv)
    guard case .spirv(let words) = spirv else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.spirv(words))
}

@Test func glslStandaloneConstructorsEvaluateTheirArguments() throws {
    let input = ShaderInput.glsl("""
        #version 440 core
        layout(local_size_x = 1) in;
        float value() { return 1.0; }
        void main() {
            vec4(value());
            float[1](value());
        }
        """, stage: .compute, defines: [:])
    let module = try IndependentShaderCompiler().parseAndValidate(input).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    #expect(main.statements.values.filter {
        if case .call = $0.kind { return true }
        return false
    }.count == 2)
}

@Test(arguments: [
    "float[2](1.0)",
    "float[2](1.0, 2.0, 3.0)",
    "float[2](true, 2.0)",
    "float[1](1.0)",
])
func glslRuntimeArrayConstructorRejectsInvalidInitializers(_ initializer: String) throws {
    let input = ShaderInput.glsl("""
        #version 440 core
        layout(local_size_x = 1) in;
        void main() { float values[2] = \(initializer); }
        """, stage: .compute, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Expected a type mismatch")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.map(\.code) == [.typeMismatch])
    }
}
