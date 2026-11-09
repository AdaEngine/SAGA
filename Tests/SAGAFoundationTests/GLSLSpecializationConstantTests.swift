import SAGACompiler
import SAGAShaderIR
import Testing

private let glslSpecializationSource = ShaderInput.glsl("""
    #version 450
    layout(constant_id = 0) const bool enabled = true;
    layout(constant_id = 1) const int bias = 42;
    layout(constant_id = 2) const uint count = 10u;
    layout(constant_id = 3) const float scale = 3.25;
    layout(constant_id = 4) const float offset = float(bias) + scale;
    layout(location = 0) out vec4 color;

    void main() {
        float result = 0.0;
        if (enabled) { result += offset; }
        result += float(count) * scale;
        color = vec4(result, 0.0, 0.0, 1.0);
    }
    """, stage: .fragment, defines: [:])

@Test func glslSpecializationConstantsPreserveTypedIDsAndDependencies() throws {
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(glslSpecializationSource).module
    #expect(module.overrides.values.map(\.id) == [0, 1, 2, 3, 4])
    #expect(try module.overrides.values.map { try module.types.value(for: $0.type).kind }
        == [.scalar(.bool), .scalar(.i32), .scalar(.u32), .scalar(.f32), .scalar(.f32)])
    let initializer = try #require(module.overrides.values.last?.initializer)
    let expression = try module.globalExpressions.value(for: initializer)
    guard case .binary(.add, let left, let right) = expression.kind,
          case .cast(let source, .f32) = try module.globalExpressions.value(for: left).kind,
          case .override(let bias) = try module.globalExpressions.value(for: source).kind,
          case .override(let scale) = try module.globalExpressions.value(for: right).kind else {
        Issue.record("Expected a typed expression referencing two overrides")
        return
    }
    #expect(try module.overrides.value(for: bias).id == 1)
    #expect(try module.overrides.value(for: scale).id == 3)
}

@Test func glslSpecializationConstantsApplyDefaultsAndSuppliedIDs() throws {
    let compiler = IndependentShaderCompiler()
    let defaults = try compiler.compileRenderStage(glslSpecializationSource, entryPoint: "main", to: .wgsl)
    let supplied = try compiler.compileRenderStage(glslSpecializationSource, entryPoint: "main", to: .wgsl,
                                                   constants: ["1": 17, "3": 6.5])
    guard case .text(.wgsl, let defaultSource) = defaults.artifact,
          case .text(.wgsl, let suppliedSource) = supplied.artifact else {
        throw ShaderFailure.incompatibleArtifact
    }
    #expect(defaultSource.contains("45.25"))
    #expect(suppliedSource.contains("23.5"))
    _ = try compiler.parseAndValidate(.wgsl(defaultSource))
    _ = try compiler.parseAndValidate(.wgsl(suppliedSource))
}

@Test(arguments: [
    ("layout(constant_id = 0) const vec2 value = vec2(1.0);", ShaderDiagnosticCode.typeMismatch),
    ("layout(constant_id = 0) const float value[2] = { 1.0, 2.0 };", .typeMismatch),
    ("layout(constant_id = 65536) const uint value = 1u;", .invalidBinding),
    ("layout(constant_id = 0) const uint a = 1u; layout(constant_id = 0) const uint b = 2u;", .duplicateName),
    ("layout(constant_id = 0) const uint a = 1u; const uint a = 2u;", .duplicateName),
    ("layout(constant_id = 0) const float a = 1.0; const vec2 value = vec2(a);", .unsupportedConstruct),
])
func glslSpecializationConstantsDiagnoseUnsupportedDeclarations(source: String, code: ShaderDiagnosticCode) throws {
    let input = ShaderInput.glsl("#version 450\n\(source)\nvoid main() {}", stage: .fragment, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Invalid specialization constant accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}
