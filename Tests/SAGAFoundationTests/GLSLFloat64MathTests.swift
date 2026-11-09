import SAGACompiler
import SAGAShaderIR
import Testing

private let doubleMatrixShader = ShaderInput.glsl("""
    #version 450
    layout(location = 0) out vec4 color;
    void main() {
        dmat2x3 matrix = dmat2x3(dvec3(1.0000000000000002LF, 2.0lf, 3.0lf), dvec3(4.0lf, 5.0lf, 6.0lf));
        dmat3x2 transposed = transpose(matrix);
        dmat2 identity = dmat2(1.0lf);
        dmat2 square = mat2(1.0) * (transposed * matrix);
        if (identity == square) { square = identity; }
        double norm = length(square[0]);
        double scale = ldexp(determinant(square), 1);
        dvec2 adjusted = smoothstep(0.0, 100.0, ceil(square[1]));
        color = vec4(float(norm), float(scale), vec2(adjusted));
    }
    """, stage: .fragment, defines: [:])

@Test func glslFloat64MatrixAndMathPreserveTypesAcrossNativeOutputs() throws {
    let compiler = IndependentShaderCompiler()
    let capabilities: ShaderCapabilities = [.float64]
    let validated = try compiler.parseAndValidate(doubleMatrixShader, capabilities: capabilities)
    #expect(validated.module.types.values.contains {
        $0.kind == .matrix(columns: .two, rows: .three, scalar: .f64)
    })
    let output = try compiler.compileRenderStage(doubleMatrixShader, entryPoint: "main", to: .glsl)
    guard case .text(.glsl, let source) = output.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("dmat2x3"))
    #expect(source.contains("transpose("))
    #expect(source.contains("ldexp("))
    guard case .spirv(let words) = try compiler.compile(doubleMatrixShader, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let roundTrip = try compiler.parseAndValidate(.spirv(words), capabilities: [.float64, .shaderInt64])
    #expect(roundTrip.module.globalExpressions.values.contains { $0.kind == .literal(.f64(1.0.nextUp)) })
}

@Test func glslFloat64TranscendentalsRequireAnAvailableOverload() {
    let input = ShaderInput.glsl("""
        #version 450
        void main() { double value = sin(0.5lf); }
        """, stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(input, capabilities: [.float64])
    }
}
