import Testing
import SAGACompiler
import SAGAShaderIR
import SAGAValidation
import SAGATypes
import SAGACore
import SAGAMetal

private let decompositionShader = """
    const abstractParts = frexp(-0.125);
    const_assert abstractParts.fract == -0.5 && abstractParts.exp == -2;
    const_assert frexp(vec3(8, -1, 3)).exp.z == 2;
    const_assert modf(vec4f(-1, -0.25, 2.5, 3.75)).fract.w == 0.75f;
    const_assert modf(1e300).whole == 1e300;
    const_assert modf(-3.25f).fract == -0.25f;
    const_assert modf(vec2f(-3.25, 2.75)).whole.y == 2.0f;
    const_assert frexp(vec2f(8, -0.125)).exp.y == -2i;
    @group(0) @binding(0) var<storage, read> input: array<vec4f, 1>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 4>;
    @compute @workgroup_size(1) fn run() {
        const parts = modf(vec3(-1.75, 5.25, 3.0));
        const_assert parts.whole.x == -1.0;
        let x = input[0];
        let fraction = frexp(x);
        let whole = modf(x.zw);
        var scalar = frexp(x.y);
        let huge = frexp(1e300);
        scalar.exp += huge.exp - 997i;
        let triple = modf(x.xyz);
        output[0] = fraction.fract;
        output[1] = vec4f(fraction.exp);
        output[2] = vec4f(whole.fract, whole.whole);
        output[3] = vec4f(scalar.fract, f32(scalar.exp), triple.fract.z, parts.fract.y);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsDecompositionBuiltins(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(decompositionShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(source.contains("modf("))
    if format == .hlsl {
        #expect(source.contains("firstbithigh(") && !source.contains("frexp("))
    } else {
        #expect(source.contains("frexp("))
    }
    if format == .wgsl {
        #expect(source.contains(".fract") && source.contains(".exp") && source.contains(".whole"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [
    "const x = modf(vec2u(1));",
    "const x = frexp(1.0, 2.0);",
    "const x = modf(1.0).exp;",
    "const x = true || bool(modf(1.0));",
    "@compute @workgroup_size(1) fn run() { var x = 1u; let parts = frexp(x); }",
])
func rejectsInvalidDecompositionBuiltins(source: String) throws {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsDecompositionInFragmentHelper(format: ShaderOutputFormat) throws {
    let source = """
        fn split(x: vec2f) -> vec4f {
            let parts = frexp(x);
            return vec4f(parts.fract, vec2f(parts.exp));
        }
        @fragment fn fs(@location(0) x: vec2f) -> @location(0) vec4f { return split(x); }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: format)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected shader source"); return }
    if format == .hlsl {
        #expect(body.contains("firstbithigh(") && !body.contains("frexp("))
    } else {
        #expect(body.contains("frexp("))
    }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(body)) }
}

@Test(arguments: [4, 8, 16])
func validatesDecompositionResultLayout(secondOffset: UInt32) throws {
    let lane = try IRHandle<IRType>(index: 0)
    let exponent = try IRHandle<IRType>(index: 1)
    let result = try IRHandle<IRType>(index: 2)
    let members: [IRStructMember] = [
        .init(name: "fract", type: lane, offset: 0),
        .init(name: "exp", type: exponent, offset: secondOffset),
    ]
    let types = IRTable([
        IRType(kind: .scalar(.f32)), IRType(kind: .scalar(.i32)),
        IRType(kind: .structure(members: members, span: secondOffset + 4)),
    ])
    let module = IRModule(types: types, specialTypes: .init(predeclared: [
        .init(kind: .frexpResult(size: nil, scalar: .f32), type: result),
    ]))
    if secondOffset == 4 {
        try ShaderSemanticValidator().validate(module)
    } else {
        #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(module) }
    }
}

private let ldexpShader = """
    const split = frexp(-3.25);
    const_assert ldexp(split.fract, split.exp) == -3.25;
    const_assert ldexp(0.5, 1024) == 0x1p1023;
    const_assert ldexp(0.5f, 128) == 0x1p127f;
    const_assert ldexp(8, -2) == 2.0;
    const_assert bitcast<u32>(ldexp(0.5, 2i)) == 0x40000000u;
    const_assert ldexp(vec3(2, 4, 8), vec3i(-1, 0, 2)).z == 32.0f;
    const_assert ldexp(1.0, -9223372036854775807 - 1) == 0.0;
    const_assert bitcast<u32>(ldexp(1.5f, -149i)) == 2u;
    @group(0) @binding(0) var<storage, read> input: array<vec4f, 1>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 2>;
    @compute @workgroup_size(1) fn run() {
        let x = input[0];
        let split = frexp(x);
        output[0] = ldexp(split.fract, split.exp);
        output[1] = vec4f(ldexp(x.y, -4), ldexp(x.z, 3i), ldexp(0.5, split.exp.x), ldexp(0.5f, 128));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsLdexpWithSignedExponent(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(ldexpShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(source.contains("ldexp("))
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(source)) }
}

@Test(arguments: [
    ("const x = ldexp(0.5f, 129);", ShaderDiagnosticCode.invalidType),
    ("const x = ldexp(0.5, 1025);", .invalidType),
    ("const x = ldexp(1.0f, 128);", .invalidType),
    ("const x = ldexp(1.0, 1024);", .invalidType),
    ("const x = ldexp(vec2f(1), vec2i(-2, 129));", .invalidType),
    ("const x = true || ldexp(0.0f, 129) > 0.0f;", .invalidType),
    ("const x = ldexp(vec2f(1), 0);", .typeMismatch),
    ("const x = ldexp(1i, 2);", .typeMismatch),
    ("const x = ldexp(1.0f, 2u);", .typeMismatch),
    ("const x = ldexp(1.0f, 2.0f);", .typeMismatch),
    ("@compute @workgroup_size(1) fn run() { var x = 0.5f; let y = ldexp(x, 129); }", .invalidType),
    ("@compute @workgroup_size(1) fn run() { var x = vec2f(0.5); let y = ldexp(x, vec2(128, 129)); }", .invalidType),
])
func diagnosesInvalidLdexp(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid ldexp accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

@Test func rejectsUnsignedLdexpExponentInRawIR() throws {
    let source = "fn scale(x: vec2f, y: vec2u) -> vec2f { return x; }"
    let module = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
    let original = try #require(module.functions.values.first)
    let x = try IRHandle<IRExpression>(index: 0)
    let y = try IRHandle<IRExpression>(index: 1)
    let z = try IRHandle<IRExpression>(index: 2)
    let expressions = original.expressions.values + [
        .init(type: original.arguments[0].type,
              kind: .math(function: .ldexp, argument: x, argument1: y, argument2: nil, argument3: nil)),
    ]
    let body = IRFunction(arguments: original.arguments, result: original.result,
                          expressions: IRTable(expressions),
                          statements: IRTable([.init(kind: .emit([z])), .init(kind: .returnValue(z))]),
                          blocks: IRTable([.init(statements: [try IRHandle(index: 0), try IRHandle(index: 1)])]),
                          body: original.body)
    let malformed = IRModule(types: module.types, functions: IRTable([body]))
    do {
        try ShaderSemanticValidator().validate(malformed)
        Issue.record("Unsigned exponent accepted in raw ldexp IR")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func recomposesFrexpWithLdexpOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(ldexpShader), to: .msl)
    let values: [Float] = [.greatestFiniteMagnitude, -3.25, 0.125, .leastNormalMagnitude]
    let expected = values + [-0.203125, 1, 0x1p127, 0x1p127]
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: values.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        ])
        #expect(try await ticket.wait().first == expected.map(\.bitPattern))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

private let decompositionInput: [Float] = [8, -0.125, -3.25, 2.75]
private let expectedDecomposition: [Float] = [
    0.5, -0.5, -0.8125, 0.6875, 4, -2, 2, 2,
    -0.25, 0.75, -3, 2, -0.5, -2, -0.25, 0.25,
]

#if canImport(Metal)
@Test func evaluatesDecompositionBuiltinsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(decompositionShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: decompositionInput.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        ])
        let values = try #require(try await ticket.wait().first).map { Float(bitPattern: $0) }
        #expect(values == expectedDecomposition)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
