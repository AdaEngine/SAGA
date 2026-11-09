import SAGACompiler
import SAGAShaderIR
import Testing

private let glslOverloads = ShaderInput.glsl("""
    #version 450 core
    precision highp float;
    layout(location = 0) out vec4 color;

    float exact(float value) { return value + 10.0; }
    int exact(int value) { return value + 2; }
    float implicit(float value) { return value + 20.0; }
    int implicit(int value) { return value + 3; }
    vec3 adapt(float value) { return vec3(value); }
    vec3 adapt(vec2 value) { return vec3(value, 0.0); }
    vec3 adapt(vec3 value) { return value + vec3(30.0); }
    vec3 adapt(vec4 value) { return value.xyz; }
    void observe(int value) { color.a = float(value); }
    vec3 observe(float value) { return vec3(value); }

    void main() {
        color = vec4(float(exact(1)), implicit(1u), adapt(ivec3(2)).x, 1.0);
        observe(2);
        observe(1.0).x + 2.0;
    }
    """, stage: .fragment, defines: [:])

@Test func glslFunctionOverloadsSelectExactAndShapeCompatibleTargets() throws {
    let module = try IndependentShaderCompiler().parseAndValidate(glslOverloads).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let calls = try main.statements.values.compactMap { statement -> (parameter: IRTypeKind, result: Bool)? in
        guard case .call(let handle, let arguments, let result) = statement.kind else { return nil }
        let function = try module.functions.value(for: handle)
        let parameter = try #require(function.arguments.first)
        let argument = try #require(arguments.first)
        #expect(try main.expressions.value(for: argument).type == parameter.type)
        return (try module.types.value(for: parameter.type).kind, result != nil)
    }
    #expect(calls.map(\.parameter) == [.scalar(.i32), .scalar(.f32), .vector(size: .three, scalar: .f32), .scalar(.i32), .scalar(.f32)])
    #expect(calls.map(\.result) == [true, true, true, false, true])
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func glslFunctionOverloadsReachTextTargets(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileRenderStage(glslOverloads, entryPoint: "main", to: format)
    #expect(artifact.colorOutputComponentCounts == [0: 4])
    if format == .wgsl, case .text(.wgsl, let text) = artifact.artifact {
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

@Test func glslFunctionOverloadsPreferIntegerToFloatOverDouble() throws {
    let source = ShaderInput.glsl("""
        #version 450 core
        void choose(double value) {}
        void choose(float value) {}
        void main() { choose(1u); }
        """, stage: .fragment, defines: [:])
    let module = try IndependentShaderCompiler().parseAndValidate(source, capabilities: [.float64]).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let call = try #require(main.statements.values.first { if case .call = $0.kind { true } else { false } })
    guard case .call(let handle, _, _) = call.kind else { throw ShaderFailure.invalidHandle }
    let parameter = try #require(module.functions.value(for: handle).arguments.first)
    #expect(try module.types.value(for: parameter.type).kind == .scalar(.f32))
}

@Test(arguments: [
    ("void pick(float a, int b) {} void pick(int a, float b) {} void main() { pick(1, 1); }", ShaderDiagnosticCode.typeMismatch),
    ("void pick(uint a) {} void pick(float a) {} void main() { pick(1); }", .typeMismatch),
    ("void pick(int a) {} void main() { pick(1u); }", .typeMismatch),
    ("void pick(vec4 a) {} void main() { pick(ivec3(1)); }", .typeMismatch),
    ("int pick(float a) { return 1; } float pick(float b) { return 2.0; } void main() {}", .duplicateName),
])
func glslFunctionOverloadsRejectAmbiguousMissingOrDuplicateSignatures(_ body: String, _ code: ShaderDiagnosticCode) throws {
    let source = ShaderInput.glsl("#version 450 core\n" + body, stage: .fragment, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source)
        Issue.record("Invalid GLSL overload accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == code)
    }
}
