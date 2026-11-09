import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

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
    if format == .wgsl {
        let web = try compiler.compileWebGPUCompute(glslHalfBufferInput)
        guard case .text(.wgsl, let source) = web.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
    }
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
    let module = try compiler.parseAndValidate(input, capabilities: [.shaderFloat16]).module
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
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(input, capabilities: [.shaderFloat16])
    }
}

#if canImport(Metal)
@Test func glslHalfUniformArithmeticExecutesOnMetal() async throws {
    let input = ShaderInput.glsl("""
        #version 460
        #extension GL_AMD_gpu_shader_half_float : enable
        layout(binding=0) uniform A { float16_t source; };
        layout(binding=1) buffer B { float result; };
        void main() { result = float(source + 0.25HF); }
        """, stage: .compute, defines: [:])
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let actual = try await device.submitCommands([
            .write(buffer: source, offset: 0, words: [0x00003e00, 0, 0, 0]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): source, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 4),
            .read(buffer: read, byteCount: 4),
        ]).wait()
        #expect(actual == [[Float(1.75).bitPattern]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
