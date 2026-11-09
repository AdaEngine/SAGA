import SAGACompiler
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation
import Testing

// naga/tests/in/wgsl/per-vertex.wgsl at the pinned Naga revision.
private let pinnedPerVertexShader = """
    enable wgpu_per_vertex;

    @fragment
    fn fs_main(@location(0) @interpolate(per_vertex) v: array<f32, 3>) -> @location(0) vec4<f32> {
        return vec4(v[0], v[1], v[2], 1.0);
    }
    """

@Test func validatesPinnedWGSLPerVertexInterface() throws {
    let module = try IndependentShaderCompiler().parseAndValidate(
        .wgsl(pinnedPerVertexShader), capabilities: [.perVertex]).module
    let entry = try #require(module.entryPoints.first)
    let function = try module.functions.value(for: entry.function)
    let argument = try #require(function.arguments.first)
    #expect(argument.binding == .location(
        index: 0, interpolation: .perVertex, sampling: nil, blendSource: nil, perPrimitive: false))
    guard case .array(let element, .constant(3), _) = try module.types.value(for: argument.type).kind else {
        Issue.record("Expected three per-vertex values")
        return
    }
    #expect(try module.types.value(for: element).kind == .scalar(.f32))
}

@Test func validatesWGSLPerVertexIntegerVectorStructMember() throws {
    let source = """
        enable wgpu_per_vertex;
        struct Input {
            @location(0) @interpolate(per_vertex) indices: array<vec2u, 3>,
            @location(1) scale: f32,
        }
        @fragment fn fs(input: Input) -> @location(0) vec4u {
            return vec4u(input.indices[2], u32(input.scale), 1u);
        }
        """
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.perVertex])
}

@Test func normalizesExplicitCenterWGSLPerVertexSampling() throws {
    let source = """
        enable wgpu_per_vertex;
        @fragment fn fs(@location(0) @interpolate(per_vertex, center) v: array<f32, 3>) -> @location(0) f32 {
            return v[0];
        }
        """
    let module = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.perVertex]).module
    let entry = try #require(module.entryPoints.first)
    let function = try module.functions.value(for: entry.function)
    #expect(function.arguments.first?.binding == .location(
        index: 0, interpolation: .perVertex, sampling: nil, blendSource: nil, perPrimitive: false))
}

@Test func wgslPerVertexRequiresDirectiveAndCapability() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(pinnedPerVertexShader)) }
    let withoutDirective = String(pinnedPerVertexShader.dropFirst("enable wgpu_per_vertex;".count))
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl(withoutDirective), capabilities: [.perVertex])
    }
    let rawModule = try WGSLFrontend(options: .init(capabilities: [.perVertex])).parse(.wgsl(pinnedPerVertexShader))
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(rawModule) }
}

@Test(arguments: [
    "f32", "array<f32, 2>", "array<f32, 4>", "array<f32>", "array<bool, 3>",
    "array<mat2x2f, 3>", "array<array<f32, 2>, 3>",
])
func rejectsInvalidWGSLPerVertexValueType(type: String) {
    let source = """
        enable wgpu_per_vertex;
        @fragment fn fs(@location(0) @interpolate(per_vertex) v: \(type)) -> @location(0) vec4f {
            return vec4f(1.0);
        }
        """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.perVertex])
    }
}

@Test(arguments: [
    """
    @vertex fn vs(@location(0) @interpolate(per_vertex) v: array<f32, 3>) -> @builtin(position) vec4f {
        return vec4f(v[0]);
    }
    """,
    """
    struct Output {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(per_vertex) values: array<f32, 3>,
    }
    @vertex fn vs() -> Output { return Output(); }
    """,
    """
    @fragment fn fs() -> @location(0) @interpolate(per_vertex) array<f32, 3> {
        return array(0.0, 1.0, 2.0);
    }
    """,
    "@compute @workgroup_size(1) fn cs(@location(0) @interpolate(per_vertex) v: array<f32, 3>) {}",
])
func rejectsWGSLPerVertexOutsideFragmentInputs(entry: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(
            .wgsl("enable wgpu_per_vertex;\n" + entry), capabilities: [.perVertex])
    }
}

@Test(arguments: ["centroid", "sample", "first", "either"])
func rejectsNonCenterWGSLPerVertexSampling(sampling: String) {
    let source = """
        enable wgpu_per_vertex;
        @fragment fn fs(@location(0) @interpolate(per_vertex, \(sampling)) v: array<f32, 3>) -> @location(0) f32 {
            return v[0];
        }
        """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.perVertex])
    }
}
