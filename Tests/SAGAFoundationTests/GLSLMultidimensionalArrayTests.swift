import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslPrivateMultidimensionalArrays() throws {
    let input = ShaderInput.glsl("""
        #version 450
        const int rows = 2;
        float grid[rows][3u];
        float deep[2][2][2][2][2][2][2];
        layout(location = 0) out vec4 color;

        void main() {
            grid[1][2] = 1.0;
            deep[1][0][1][0][1][0][1] = grid[1][2];
            color = vec4(deep[1][0][1][0][1][0][1]);
        }
        """, stage: .fragment, defines: [:])
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input).module

    let grid = try #require(module.globalVariables.values.first { $0.name == "grid" })
    guard case .array(let rowType, .constant(2), 12) = try module.types.value(for: grid.type).kind,
          case .array(let elementType, .constant(3), 4) = try module.types.value(for: rowType).kind else {
        Issue.record("Expected two rows of three tightly packed floats")
        return
    }
    #expect(try module.types.value(for: elementType).kind == .scalar(.f32))

    let deep = try #require(module.globalVariables.values.first { $0.name == "deep" })
    var current = deep.type
    for expectedStride: UInt32 in [256, 128, 64, 32, 16, 8, 4] {
        guard case .array(let child, .constant(2), let stride) = try module.types.value(for: current).kind else {
            Issue.record("Expected seven nested arrays of length two")
            return
        }
        #expect(stride == expectedStride)
        current = child
    }
    #expect(try module.types.value(for: current).kind == .scalar(.f32))

    for format in [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl] {
        _ = try compiler.compile(input, to: format)
    }
    let spirv = try compiler.compile(input, to: .spirv)
    guard case .spirv(let words) = spirv else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.spirv(words))
}

@Test(arguments: ["[2][]", "[][2]", "[2][1073741824]", "[1073741824][2]"])
func glslPrivateMultidimensionalArraysRejectUnsizedOrOverflowingDimensions(_ suffix: String) throws {
    let input = ShaderInput.glsl("""
        #version 450
        float values\(suffix);
        void main() {}
        """, stage: .fragment, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Expected an invalid array type")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.map(\.code) == [.invalidType])
    }
}
