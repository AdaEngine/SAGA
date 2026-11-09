import SAGACompiler
import SAGAFrontends
import SAGAShaderIR
import Testing

private let glslParameterQualifiers = ShaderInput.glsl("""
    #version 450 core
    layout(location = 0) out vec4 color;

    uint update(inout uint value, in const int index, out uvec2 lanes) {
        value += uint(index);
        lanes = uvec2(2u, 3u);
        if (index > 0) { return value; }
        return 0u;
    }
    void alias(inout uint first, inout uint second) {
        first += 1u;
        second += 2u;
    }
    void inputOnly(in uint value) { value += 4u; }
    void arrayOutput(inout uint[2u] values) { values[1] = 5u; }
    void choose(out uint value, int marker) { value = uint(marker); }
    void choose(float value, bool marker) {}

    void main() {
        uint values[2];
        values[0] = 7u;
        int index = 0;
        vec3 lanes = vec3(1.0);
        uint result = update(values[index], index++, lanes.xz);
        alias(values[0], values[0]);
        inputOnly(values[0]);
        arrayOutput(values);
        choose(lanes.y, 9);
        color = vec4(lanes, float(result + values[0]));
    }
    """, stage: .fragment, defines: [:])

@Test func glslOutputParametersUseSeparateCallTemporariesAndWriteBackAfterReturn() throws {
    let module = try IndependentShaderCompiler().parseAndValidate(glslParameterQualifiers).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let statements = try main.blocks.value(for: main.body).statements.map { try main.statements.value(for: $0).kind }
    let update = try #require(statements.firstIndex { statement in
        if case .call(_, _, let result) = statement { return result != nil }
        return false
    })
    guard case .call(_, let arguments, let result) = statements[update] else {
        throw ShaderFailure.invalidHandle
    }
    let temporary = arguments[0]
    let lanes = arguments[2]
    let copyIn = try #require(statements[..<update].lastIndex {
        if case .store(let pointer, _) = $0 { return pointer == temporary }
        return false
    })
    // The array value is copied before evaluation increments the index.
    #expect(statements[(copyIn + 1)..<update].contains {
        if case .store(let pointer, _) = $0 { return pointer != temporary }
        return false
    })
    #expect(!statements[..<update].contains {
        if case .store(let pointer, _) = $0 { return pointer == lanes }
        return false
    })
    let copyBack = try #require(statements[(update + 1)...].firstIndex {
        if case .store(_, let value) = $0,
           case .load(let pointer) = try? main.expressions.value(for: value).kind { return pointer == temporary }
        return false
    })
    guard case .store(_, let input) = statements[copyIn],
          case .load(let source) = try main.expressions.value(for: input).kind,
          case .store(let destination, _) = statements[copyBack] else {
        throw ShaderFailure.invalidHandle
    }
    #expect(source == destination)
    let resultStore = try #require(statements[(update + 1)...].firstIndex {
        if case .store(_, let value) = $0 { return value == result }
        return false
    })
    #expect(copyBack < resultStore)
    let aliasArguments = try #require(try statements.compactMap { statement -> [IRHandle<IRExpression>]? in
        guard case .call(let handle, let arguments, _) = statement,
              try module.functions.value(for: handle).name == "alias" else { return nil }
        return arguments
    }.first)
    #expect(Set(aliasArguments).count == 2)
    let aliasLocals = try aliasArguments.compactMap { argument -> IRHandle<IRLocalVariable>? in
        guard case .localVariable(let local) = try main.expressions.value(for: argument).kind else { return nil }
        return local
    }
    #expect(Set(aliasLocals).count == 2)
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func glslParameterQualifiersReachTextTargets(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileRenderStage(glslParameterQualifiers, entryPoint: "main", to: format)
    #expect(artifact.colorOutputComponentCounts == [0: 4])
    if format == .wgsl, case .text(.wgsl, let text) = artifact.artifact {
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

@Test func glslOutArgumentDoesNotLoadWriteonlyStorage() throws {
    let source = ShaderInput.glsl("""
        #version 450 core
        layout(local_size_x = 1) in;
        layout(std430, binding = 0) writeonly buffer Output { vec4 value; } outputBuffer;
        void set(out vec2 value) { value = vec2(2.0); }
        void main() { set(outputBuffer.value.xz); }
        """, stage: .compute, defines: [:])
    let module = try GLSLFrontend(options: .init(stage: .compute)).parse(source)
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let loads = try main.expressions.values.compactMap { expression -> IRTypeKind? in
        guard case .load(let pointer) = expression.kind else { return nil }
        return try module.types.value(for: main.expressions.value(for: pointer).type).kind
    }
    let load = try #require(loads.first)
    guard case .pointer(_, .function) = load else {
        Issue.record("Expected a load from the output temporary")
        return
    }
    #expect(loads.count == 1)
}

@Test(arguments: [
    ("void set(const int a) { a = 1; } void main() {}", ShaderDiagnosticCode.invalidAddressSpace),
    ("void set(const out int a) {} void main() {}", .invalidType),
    ("void set(in out int a) {} void main() {}", .invalidType),
    ("void set(out int a) { a = 1; } void main() { set(0); }", .invalidAddressSpace),
    ("void set(out vec2 a) {} void main() { vec3 v; set(v.xx); }", .invalidAddressSpace),
    ("void set(inout int a) {} void main() { float v; set(v); }", .typeMismatch),
    ("void set(int a) {} void set(out int a) {} void main() {}", .duplicateName),
])
func glslParameterQualifiersRejectInvalidUses(_ body: String, _ code: ShaderDiagnosticCode) throws {
    let source = ShaderInput.glsl("#version 450 core\n" + body, stage: .fragment, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source)
        Issue.record("Invalid GLSL parameter use accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == code)
    }
}
