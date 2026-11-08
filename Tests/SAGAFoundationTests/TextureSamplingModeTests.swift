import Testing
import SAGACompiler
import SAGAShaderIR
import SAGAValidation

private let bindings = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var state: sampler;
    """

@Test(arguments: [
    "@compute @workgroup_size(1) fn run() { let color = sampleAt(vec2f(0.5)); }",
    "@vertex fn run() -> @builtin(position) vec4f { return sampleAt(vec2f(0.5)); }",
])
func implicitSampleIsFragmentOnly(entry: String) throws {
    let source = bindings + "\n" + """
        fn sampleAt(uv: vec2f) -> vec4f { return textureSample(image, state, uv); }
        \(entry)
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Implicit sampling accepted outside fragment")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .invalidControlFlow)
    }
}

@Test(arguments: [
    """
    @fragment fn run(@location(0) uv: vec2f) -> @location(0) vec4f {
        if uv.x > 0.5 { return textureSample(image, state, uv); }
        return vec4f(0.0);
    }
    """,
    """
    fn read(uv: vec2f) -> vec4f { return textureSampleBias(image, state, uv, 0.0); }
    fn forwarded(uv: vec2f) -> vec4f { return read(uv); }
    @fragment fn run(@location(0) uv: vec2f) -> @location(0) vec4f {
        if uv.x > 0.5 { return forwarded(uv); }
        return vec4f(0.0);
    }
    """,
    """
    @fragment fn run() -> @location(0) vec4f {
        let first = textureSample(image, state, vec2f(0.5));
        if first.r > 0.5 { return textureSample(image, state, vec2f(0.5)); }
        return vec4f(0.0);
    }
    """,
])
func implicitSampleRequiresUniformControlFlow(entry: String) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(bindings + "\n" + entry))
        Issue.record("Implicit sampling accepted in divergent control flow")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .nonUniformControlFlow)
    }
}

@Test func discardReconvergesBeforeImplicitTextureSampling() throws {
    let source = bindings + "\n" + """
        @fragment fn run(@location(0) uv: vec2f) -> @location(0) vec4f {
            if uv.x > 0.5 { discard; }
            return textureSample(image, state, uv);
        }
        """
    let compiler = IndependentShaderCompiler()
    for format in [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl] {
        _ = try compiler.compileRenderStage(.wgsl(source), entryPoint: "run", to: format)
    }
}

@Test func explicitGradientsWorkInDivergentComputeControlFlow() throws {
    let source = bindings + "\n" + """
        @group(0) @binding(2) var<storage, read_write> out: vec4f;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 0u {
                out = textureSampleGrad(image, state, vec2f(0.5), vec2f(0.25, 0.0), vec2f(0.0, 0.25));
            }
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    guard case .text(_, let code) = shader.artifact else { Issue.record("Expected source"); return }
    #expect(code.contains("textureSampleGrad("))
    _ = try compiler.parseAndValidate(.wgsl(code))
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsBaseClampSamplingInDivergentCompute(format: ShaderOutputFormat) throws {
    let source = bindings + "\n" + """
        @group(0) @binding(2) var<storage, read_write> out: vec4f;
        fn sampleAt(uv: vec2f) -> vec4f { return textureSampleBaseClampToEdge(image, state, uv,); }
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 0u { out = sampleAt(vec2f(-0.25, 1.25)); }
        }
        """
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(.wgsl(source)).module
    let sample = try #require(module.functions.values.flatMap(\.expressions.values).compactMap { expression in
        if case .imageSample(let sample) = expression.kind { return sample }
        return nil
    }.first)
    #expect(sample.clampToEdge && sample.level == .zero)
    let artifact = try compiler.compileCompute(.wgsl(source), to: format)
    guard case .text(_, let code) = artifact.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(code.contains("textureSampleBaseClampToEdge("))
        _ = try compiler.parseAndValidate(.wgsl(code))
    case .msl:
        #expect(code.contains(".get_width(0u)") && code.contains("clamp(") && code.contains("level(0.0f)"))
    case .hlsl:
        #expect(code.contains(".GetDimensions(0u") && code.contains("clamp(") && code.contains(".SampleLevel("))
    case .glsl:
        #expect(code.contains("textureSize(saga_pair_0_1, int(0u))"))
        #expect(code.contains("clamp(") && code.contains("textureLod(saga_pair_0_1"))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    ("image, state, vec2f(0.5), vec2i(1, 0)", ShaderDiagnosticCode.typeMismatch),
    ("image, comparison, vec2f(0.5)", ShaderDiagnosticCode.typeMismatch),
    ("depth, state, vec2f(0.5)", ShaderDiagnosticCode.typeMismatch),
    ("image, state, vec3f(0.5)", ShaderDiagnosticCode.typeMismatch),
])
func rejectsInvalidBaseClampSampling(arguments: String, code: ShaderDiagnosticCode) throws {
    let source = bindings + "\n" + """
        @group(0) @binding(2) var depth: texture_depth_2d;
        @group(0) @binding(3) var comparison: sampler_comparison;
        @compute @workgroup_size(1) fn run() { let value = textureSampleBaseClampToEdge(\(arguments)); }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid base clamp sampling accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

@Test func scalarUniformLoadCanControlImplicitSamplingAfterWGSLRoundTrip() throws {
    let source = bindings + "\n" + """
        @group(0) @binding(2) var<uniform> flag: f32;
        @fragment fn run(@location(0) uv: vec2f) -> @location(0) vec4f {
            if flag > 0.0 { return textureSample(image, state, uv); }
            return vec4f(0.0);
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(source), entryPoint: "run", to: .wgsl)
    guard case .text(_, let code) = shader.artifact else { Issue.record("Expected source"); return }
    _ = try compiler.parseAndValidate(.wgsl(code))
}

@Test(arguments: [
    ("vec2i(-9, 0)", ShaderDiagnosticCode.invalidType),
    ("vec2i(0, 8)", ShaderDiagnosticCode.invalidType),
    ("vec2i(dynamicOffset, 0)", ShaderDiagnosticCode.invalidType),
    ("vec2u(1u, 0u)", ShaderDiagnosticCode.typeMismatch),
    ("vec3i(1, 0, 0)", ShaderDiagnosticCode.typeMismatch),
])
func rejectsInvalidTextureSampleOffsets(offset: String, diagnostic: ShaderDiagnosticCode) throws {
    let source = bindings + "\n" + """
        @compute @workgroup_size(1) fn run() {
            let dynamicOffset = 1i;
            let color = textureSampleLevel(image, state, vec2f(0.5), 0.0, \(offset));
        }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid sample offset accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == diagnostic)
    }
}

@Test func foldsBoundaryTextureSampleOffsetsInWGSL() throws {
    let source = bindings + "\n" + """
        const x = -4i;
        const y = 1i;
        @compute @workgroup_size(1) fn run() {
            const offset = vec2(x + x, y + 6i);
            let color = textureSampleLevel(image, state, vec2f(0.5), 0.0, offset);
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    guard case .text(_, let code) = shader.artifact else { Issue.record("Expected source"); return }
    #expect(code.contains("-8i") && code.contains("7i"))
    _ = try compiler.parseAndValidate(.wgsl(code))
}

@Test func semanticValidatorRejectsOutOfRangeOffsetsInTypedIR() throws {
    let source = bindings + "\n" + """
        @compute @workgroup_size(1) fn run() {
            let color = textureSampleLevel(image, state, vec2f(0.5), 0.0, vec2i(0, 0));
        }
        """
    let parsed = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
    let function = try #require(parsed.functions.values.first)
    var expressions = function.expressions.values
    let offset = try #require(expressions.compactMap { expression -> IRHandle<IRExpression>? in
        if case .imageSample(let sample) = expression.kind { return sample.offset }
        return nil
    }.first)
    guard case .compose(_, let components) = expressions[offset.index].kind else {
        Issue.record("Expected a folded offset vector")
        return
    }
    let lane = try #require(components.first)
    expressions[lane.index] = .init(type: expressions[lane.index].type, kind: .literal(.i32(8)))
    let forged = IRFunction(expressions: IRTable(expressions), statements: function.statements,
                            blocks: function.blocks, body: function.body)
    let module = IRModule(types: parsed.types, globalVariables: parsed.globalVariables,
                          functions: IRTable([forged]), entryPoints: parsed.entryPoints)
    do {
        _ = try ShaderSemanticValidator().validate(module)
        Issue.record("Out-of-range typed IR offset accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .invalidType)
    }
}
