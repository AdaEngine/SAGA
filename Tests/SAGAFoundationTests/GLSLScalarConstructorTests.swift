import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslScalarConstructorsExtractFirstVectorAndMatrixComponents() throws {
    let input = ShaderInput.glsl("""
        #version 440 core
        layout(local_size_x = 1) in;
        layout(std430, binding = 0) buffer Result { int value; } outputValue;
        void main() {
            mat2x3 matrix = mat2x3(2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
            float first = float(matrix);
            outputValue.value = int(vec2(first, 8.0));
        }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let componentIndices = main.expressions.values.compactMap { expression -> UInt32? in
        guard case .accessIndex(let base, let index) = expression.kind else { return nil }
        switch module.types.values[main.expressions.values[base.index].type.index].kind {
        case .vector, .matrix: return index
        default: return nil
        }
    }
    #expect(componentIndices == [0, 0, 0])
    for format in [ShaderOutputFormat.wgsl, .msl] {
        _ = try compiler.compileCompute(input, to: format)
    }
}
