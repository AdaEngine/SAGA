import SAGACompiler
import SAGAShaderIR
import Testing

// Covers the buffer declarations in Naga's f16-glsl.comp and f16-spv.comp.
private let glslHalfBufferInput = ShaderInput.glsl("""
    #version 460
    #extension GL_AMD_gpu_shader_half_float : enable
    layout(set=0,binding=0) uniform A {
        float16_t a_1;
        f16vec2 a_vec2;
        f16vec3 a_vec3;
        f16vec4 a_vec4;
    };
    layout(set=0,binding=1) buffer B {
        float16_t b_1;
        f16vec2 b_vec2;
        f16vec3 b_vec3;
        f16vec4 b_vec4;
        f16mat2 b_mat2;
        f16mat2x3 b_mat2x3;
        f16mat2x4 b_mat2x4;
        f16mat3x2 b_mat3x2;
        f16mat3 b_mat3;
        f16mat3x4 b_mat3x4;
        f16mat4x2 b_mat4x2;
        f16mat4x3 b_mat4x3;
        f16mat4 b_mat4;
    };
    void main() {
        b_1 = a_1;
        b_vec2 = a_vec2;
        b_vec3 = a_vec3;
        b_vec4 = a_vec4;
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslHalfBufferLayoutReachesTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileCompute(glslHalfBufferInput, to: format)
    #expect(artifact.requiresShaderF16)
    #expect(artifact.uniformBindings.map(\.minimumByteSize) == [32])
    #expect(artifact.storageBindings.map(\.minimumByteSize) == [208])
    if format == .wgsl { _ = try compiler.compileWebGPUCompute(glslHalfBufferInput) }
}

@Test(arguments: ["GL_AMD_gpu_shader_half_float", "GL_EXT_shader_explicit_arithmetic_types_float16"])
func glslHalfLiteralsAndConstructors(_ name: String) throws {
    let input = ShaderInput.glsl("""
        #version 460
        #if !defined(\(name))
        #error Half extension support macro is missing
        #endif
        #extension \(name) : require
        #define HALF 1.5HF
        layout(binding=0) buffer Data { f16mat2 matrix; };
        void main() { matrix = f16mat2(f16vec2(HALF, 0.25hf), vec2(0.5, 2.0)); }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input).module
    let literals = module.functions.values.flatMap { $0.expressions.values }.compactMap { expression in
        if case .literal(.f16Bits(let bits)) = expression.kind { return bits }
        return nil
    }
    #expect(literals == [0x3e00, 0x3400])
    _ = try compiler.compileWebGPUCompute(input)
}

@Test(arguments: [
    "void main() { float16_t value; }",
    "void main() { float value = 1.0HF; }",
    "#extension GL_AMD_gpu_shader_half_float : enable\n#extension GL_AMD_gpu_shader_half_float : disable\nvoid main() { f16vec2 value; }",
    "#extension GL_AMD_gpu_shader_half_float : enable\nlayout(binding=0) uniform A { f16mat2 matrix; };\nvoid main() {}",
    "#extension GL_AMD_gpu_shader_half_float : enable\nvoid main() { float16_t value = 1.0e8HF; }",
])
func glslHalfRejectsDisabledOrUnrepresentableInput(_ body: String) {
    let input = ShaderInput.glsl("#version 460\n\(body)", stage: .compute, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}
