import SAGACompiler
import SAGAShaderIR
import Testing

private let conditionalSource = ShaderInput.glsl("""
    #version 450
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer Result { uint value; };
    bool choose(inout uint n) { n++; return n == 2u; }
    void main() {
        uint n = 1u;
        int i = 4;
        uint a = choose(n) ? i++ : n++;
        uint b = n == 2u ? 2u : false ? 1u : 3;
        vec2 v = a == 4u && n == 2u ? ivec2(5) : vec2(float(i));
        value = n + uint(i) + a + b + uint(v.x);
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslConditionalExpressionsReachTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(conditionalSource, to: format)
    if case .text(.wgsl, let source) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test func glslConditionalExpressionsKeepEffectsAndConversionsInsideEachArm() throws {
    let module = try IndependentShaderCompiler().parseAndValidate(conditionalSource).module
    let function = try #require(module.functions.values.first { $0.name == "main" })
    let body = try function.blocks.value(for: function.body)
    let statements = try body.statements.map { try function.statements.value(for: $0).kind }
    let branchIndex = try #require(statements.firstIndex { if case .ifCondition = $0 { true } else { false } })
    let conditionCalls = statements[..<branchIndex].filter { if case .call = $0 { true } else { false } }
    #expect(conditionCalls.count == 1)
    guard case .ifCondition(_, let accept, let reject) = statements[branchIndex] else { return }
    let yes = try function.blocks.value(for: accept).statements.map { try function.statements.value(for: $0).kind }
    let no = try function.blocks.value(for: reject).statements.map { try function.statements.value(for: $0).kind }
    let yesStores = yes.filter { if case .store = $0 { true } else { false } }
    let noStores = no.filter { if case .store = $0 { true } else { false } }
    #expect(yesStores.count == 2)
    #expect(noStores.count == 2)
    guard case .store(let yesTarget, let yesValue) = try #require(yesStores.last),
          case .store(let noTarget, let noValue) = try #require(noStores.last) else {
        Issue.record("Expected each arm to store its result")
        return
    }
    #expect(yesTarget == noTarget)
    let yesResult = try function.expressions.value(for: yesValue)
    let noResult = try function.expressions.value(for: noValue)
    #expect(yesResult.type == noResult.type)
    #expect(try module.types.value(for: yesResult.type).kind == .scalar(.u32))
    #expect(yes.contains { if case .emit(let values) = $0 { values.contains(yesValue) } else { false } })
    guard case .cast(_, .u32) = yesResult.kind else {
        Issue.record("Expected int arm to convert to uint")
        return
    }
}

@Test(arguments: [
    "int bad = 1 ? 2 : 3;",
    "int bad = bvec2(true) ? 2 : 3;",
    "vec2 bad = true ? vec2(1) : ivec3(2);",
    "vec2 bad = true ? vec2(1) : 2;",
    "float bad = true ? false : 2.0;",
])
func glslConditionalExpressionsRejectIncompatibleTypes(_ body: String) throws {
    let source = ShaderInput.glsl("#version 450\nvoid main() { \(body) }", stage: .compute, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source)
        Issue.record("Invalid GLSL conditional accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .typeMismatch)
    }
}
