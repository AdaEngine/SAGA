import SAGACompiler
import SAGAShaderIR
import Testing

@Test(arguments: [
    ("ivec2", "vec2(-2.5, 3.5)", IRScalar.f32, IRScalar.i32),
    ("uvec2", "vec2(2.5, 3.5)", .f32, .u32),
    ("vec2", "ivec2(-2, 3)", .i32, .f32),
    ("bvec2", "ivec2(0, -3)", .i32, .bool),
    ("ivec2", "bvec2(false, true)", .bool, .i32),
])
func glslVectorConstructorsConvertMatchingWidthOperands(
    _ vectorType: String, _ argument: String, _ source: IRScalar, _ target: IRScalar
) throws {
    let input = ShaderInput.glsl("""
        #version 450
        layout(local_size_x=1) in;
        layout(binding=0) buffer Result { vec2 result; };
        void main() {
            \(vectorType) converted = \(vectorType)(\(argument));
            result = vec2(converted);
        }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let casts = try main.expressions.values.compactMap { expression -> IRTypeKind? in
        guard case .cast(let operand, let scalar) = expression.kind, scalar == target else { return nil }
        let input = try main.expressions.value(for: operand)
        return try module.types.value(for: input.type).kind
    }
    #expect(casts.contains(.vector(size: .two, scalar: source)))
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    _ = try compiler.parseAndValidate(.spirv(words))
    for format: ShaderOutputFormat in [.wgsl, .msl, .hlsl, .glsl] {
        let shader = try compiler.compileCompute(input, to: format)
        if case .text(.wgsl, let text) = shader.artifact {
            _ = try compiler.parseAndValidate(.wgsl(text))
        }
    }
}

@Test func glslVectorConstructorsConvertMixedScalarAndVectorComponents() throws {
    let input = ShaderInput.glsl("""
        #version 450
        layout(local_size_x=1) in;
        layout(binding=0) buffer Result { ivec4 result; };
        void main() { result = ivec4(vec2(-2.5, 3.5), true, 1u); }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileWebGPUCompute(input)
    guard case .text(.wgsl, let output) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.wgsl(output))
}
