import SAGACompiler
import SAGAShaderIR
import Testing

private let glslDefaultBlockLayoutInput = ShaderInput.glsl("""
    #version 450
    layout(local_size_x=1) in;
    const int WEIGHT_COUNT = 2;
    const uint EXTENDED_COUNT = 2u;
    struct Pair {
        float values[2];
        float scale;
    };
    layout(set=0,binding=0) uniform Config {
        float bias;
        float weights[WEIGHT_COUNT];
        Pair nested;
        mat3 projection;
    };
    layout(set=0,binding=1) buffer Output {
        uint marker;
        uint[] words;
    };
    layout(std140,set=0,binding=2) buffer Extended {
        float first;
        float[EXTENDED_COUNT] values;
    } extended;
    void main() {
        words[0] = uint(bias + weights[1] + nested.scale + projection[0].x + extended.values[1]);
    }
    """, stage: .compute, defines: [:])

@Test func glslResourceBlocksUseNagaVulkanDefaultLayouts() throws {
    let module = try IndependentShaderCompiler().parseAndValidate(glslDefaultBlockLayoutInput).module
    func block(_ binding: UInt32) throws -> ([IRStructMember], UInt32) {
        let global = try #require(module.globalVariables.values.first { $0.binding == .init(group: 0, binding: binding) })
        guard case .structure(let members, let span) = try module.types.value(for: global.type).kind else {
            throw ShaderFailure.invalidHandle
        }
        return (members, span)
    }
    let (uniform, uniformSpan) = try block(0)
    #expect(uniform.map(\.offset) == [0, 16, 48, 96] && uniformSpan == 144)
    guard case .array(_, .constant(2), 16) = try module.types.value(for: uniform[1].type).kind,
          case .structure(let nested, 48) = try module.types.value(for: uniform[2].type).kind,
          case .array(_, .constant(2), 16) = try module.types.value(for: nested[0].type).kind else {
        Issue.record("Expected std140 array strides and nested structure span")
        return
    }
    #expect(nested.map(\.offset) == [0, 32])
    let (storage, storageSpan) = try block(1)
    #expect(storage.map(\.offset) == [0, 4] && storageSpan == 8)
    guard case .array(_, .dynamic, 4) = try module.types.value(for: storage[1].type).kind else {
        Issue.record("Expected std430 runtime array stride")
        return
    }
    let (extended, extendedSpan) = try block(2)
    #expect(extended.map(\.offset) == [0, 16] && extendedSpan == 48)
    guard case .array(_, .constant(2), 16) = try module.types.value(for: extended[1].type).kind else {
        Issue.record("Expected explicit std140 storage array stride")
        return
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslResourceBlockLayoutsReachTextBackends(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(glslDefaultBlockLayoutInput, to: format)
    #expect(shader.uniformBindings.map(\.minimumByteSize) == [144])
    #expect(shader.storageBindings.map(\.minimumByteSize) == [8, 48])
    #expect(shader.storageBindings.first?.runtimeArrayOffset == 4)
    #expect(shader.storageBindings.first?.runtimeArrayStride == 4)
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [
    "layout(binding=0) uniform U { layout(offset=4) float values[2]; } u;",
    "layout(binding=0) uniform U { mat3x2 matrix; } u;",
    "layout(std430,binding=0) uniform U { float values[2]; } u;",
    "layout(std140,std430,binding=0) buffer B { uint value; } b;",
    "layout(binding=0) buffer A { uint member; }; layout(binding=1) buffer B { uint member; };",
    "const int COUNT = -1; layout(binding=0) buffer B { uint values[COUNT]; } b;",
    "const uint COUNT = 0u; layout(binding=0) buffer B { uint values[COUNT]; } b;",
])
func glslResourceBlockLayoutsRejectUnrepresentableOrConflictingDeclarations(declaration: String) {
    let input = ShaderInput.glsl("#version 450\n\(declaration)\nvoid main() {}", stage: .compute, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}
