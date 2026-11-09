import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

private let glslImplicitNumericSource = ShaderInput.glsl("""
    #version 450 core
    layout(local_size_x = 1) in;
    layout(std430, binding = 0) buffer Data { uint[] words; };
    const uint one = 1;

    float asFloat(uint x) { return x; }
    uint asUnsigned(int x) { return x; }

    void main() {
        uint inputValue = words[0];
        const uint two = 2;
        const float twoFloat = 2;
        uint accum = 1;
        accum += 2;
        accum = accum * inputValue + one;
        float a = inputValue;
        a += twoFloat;
        vec2 b = ivec2(-2, 6);
        b = b + inputValue;
        uvec2 c = ivec2(3, 5) & inputValue;
        ivec2 d = ivec2(-16, 16) >> two;
        uint shifted = inputValue << 2;
        vec2 flipped = 10 - b;
        bool compare = inputValue > 2;
        bool equal = ivec2(1, 9) == b;
        words[1] = accum;
        words[2] = uint(a);
        words[3] = uint(b.x);
        words[4] = uint(b.y);
        words[5] = c.x;
        words[6] = c.y;
        words[7] = uint(d.x);
        words[8] = uint(d.y);
        words[9] = shifted;
        words[10] = uint(flipped.x);
        words[11] = uint(flipped.y);
        words[12] = uint(asFloat(inputValue));
        words[13] = asUnsigned(-1);
        words[14] = uint(compare);
        words[15] = uint(equal);
        words[16] = 2;
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslImplicitNumericConversionsReachTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let result = try compiler.compileCompute(glslImplicitNumericSource, to: format)
    #expect(result.storageBindings.first?.runtimeArrayStride == 4)
    if format == .wgsl, case .text(.wgsl, let source) = result.artifact {
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [
    "uint u = 1u; int bad = u;",
    "float f = 1.5; uint bad = f;",
    "vec3 bad = ivec2(1);",
    "float bad = true;",
    "vec2 bad = 1;",
    "uint u = 1u; u += 1.5;",
    "int bad = 1 >> ivec2(1);",
    "int bad = 1 >> 1.0;",
    "float bad = 1.0 % 1;",
    "vec2 bad = vec2(1.0) & 1;",
    "bool bad = vec2(1.0) < vec2(2.0);",
])
func glslImplicitNumericConversionsRejectInvalidOperands(_ body: String) throws {
    let source = ShaderInput.glsl("#version 450 core\nvoid main() { \(body) }", stage: .compute, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source)
        Issue.record("Invalid GLSL conversion accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func glslImplicitNumericConversionsRunOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(glslImplicitNumericSource, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let size = 17 * UInt64(MemoryLayout<UInt32>.size)
        let words = try await device.createBuffer(.init(size: size, usage: [.storage, .copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
        let results = try await device.submitCommands([
            .write(buffer: words, offset: 0, words: [3]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): words], workgroups: .init(x: 1)),
            .copy(source: words, destination: readback, byteCount: size),
            .read(buffer: readback, byteCount: size),
        ]).wait()
        #expect(results == [[3, 10, 5, 1, 9, 3, 1, 0xffff_fffc, 4, 12, 9, 1, 3, .max, 1, 1, 2]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
