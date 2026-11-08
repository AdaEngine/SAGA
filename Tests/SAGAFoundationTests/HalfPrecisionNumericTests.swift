import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let halfNumericShader = """
    enable f16;
    const_assert fma(0x1.004p0h, 0x1.ff8p-1h, -1h) == -0x1p-20h;
    const_assert fma(60000h, 2h, -60000h) == 60000h;
    const_assert mix(vec2h(1, 2), vec2h(5, 6), 0.25).y == 3h;
    const_assert mix(vec2f(1, 2), vec2f(5, 6), 0.25).x == 2.0f;
    const_assert smoothstep(1h, 0h, 0.25h) == 0.84375h;
    const_assert step(vec2h(0), vec2h(-1, 0)).y == 1h;
    const_assert exp2(4h) == 16h;
    @group(0) @binding(0) var<storage, read> input: array<vec4h, 4>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 16>;
    @compute @workgroup_size(1) fn run() {
        let a = input[0];
        let b = input[1];
        let c = input[2];
        let d = input[3];
        output[0] = vec4f(round(a));
        output[1] = vec4f(clamp(a, vec4(-1.0), vec4(1.0)));
        output[2] = vec4f(saturate(a));
        output[3] = vec4f(fract(a));
        output[4] = vec4f(fma(b, vec4h(2), vec4h(-1)));
        output[5] = vec4f(mix(vec4h(0), b, c.x));
        output[6] = vec4f(mix(vec4h(0), b, vec4h(c.x, 0.5h, 0.75h, 1h)));
        output[7] = vec4f(f32(step(c.z, c.x)), f32(step(c.x, c.z)),
                          f32(smoothstep(c.y, b.w, c.z)), f32(smoothstep(c.z, c.w, c.x)));
        let blend = smoothstep(vec2h(c.y, c.z), vec2h(b.w, c.w), vec2h(c.z, c.x));
        output[8] = vec4f(vec2f(blend), f32(max(a.z, b.x)), f32(min(a.w, b.x)));
        output[9] = vec4f(f32(sqrt(b.w)), f32(inverseSqrt(c.y)), f32(log2(b.z)), f32(exp2(b.y)));
        output[10] = vec4f(f32(sin(b.x)), f32(cos(b.x)), f32(atan2(c.z, c.w)), f32(asinh(b.y)));
        output[11] = vec4f(f32(acosh(b.y)), f32(atanh(b.x)), f32(degrees(atan2(c.z, c.z))), f32(radians(d.x)));
        output[12] = vec4f(f32(exp(b.x)), f32(log(b.y)), f32(pow(b.w, b.x)), f32(sign(c.w)));
        output[13] = vec4f(f32(tan(b.x)), f32(asin(b.x)), f32(acos(b.x)), f32(atan(c.z)));
        output[14] = vec4f(f32(sinh(b.x)), f32(cosh(b.x)), f32(tanh(b.x)), f32(abs(a.x)));
        output[15] = vec4f(f32(ceil(a.x)), f32(floor(a.x)), f32(trunc(a.y)), f32(min(a.w, b.x)));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsHalfPrecisionNumericsAndInterpolation(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(halfNumericShader), to: format)
    #expect(shader.requiresShaderF16)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(body.contains(format == .hlsl ? "mad(" : "fma("))
    #expect(body.contains("step("))
    if format == .wgsl {
        #expect(body.contains("smoothstep(") && body.contains("mix("))
        _ = try compiler.parseAndValidate(.wgsl(body), capabilities: [.shaderFloat16])
    } else {
        #expect(body.contains("clamp("))
    }
}

@Test(arguments: [
    ("enable f16; const x = fma(1h, 2h);", ShaderDiagnosticCode.typeMismatch),
    ("enable f16; const x = fma(vec2h(1), vec2f(2), vec2h(3));", .typeMismatch),
    ("enable f16; const x = mix(vec2h(1), vec2h(2), 0.5f);", .typeMismatch),
    ("enable f16; const x = mix(vec2h(1), vec2h(2), vec3h(0));", .typeMismatch),
    ("enable f16; const x = step(1i, 2i);", .typeMismatch),
    ("enable f16; const x = smoothstep(1h, 1h, 0h);", .invalidType),
    ("enable f16; const x = true || smoothstep(1h, 1h, 0h) == 0h;", .invalidType),
    ("enable f16; @compute @workgroup_size(1) fn run() { var x = vec2h(0); let y = smoothstep(vec2h(1, 2), vec2h(1, 3), x); }", .invalidType),
    ("enable f16; const x = exp2(16h);", .invalidType),
    ("enable f16; const x = inverseSqrt(0h);", .invalidType),
])
func diagnosesInvalidHalfPrecisionNumerics(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
        Issue.record("Invalid half-precision numeric expression accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

#if canImport(Metal)
@Test func evaluatesHalfPrecisionNumericsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfNumericShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 256, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 256, usage: [.mapRead, .copyDestination]))
        let samples: [Double] = [-2.5, -1.5, 1.5, 2.5, 0.5, 2, 8, 9, 0.25, 4, 1, -1, 180, 0, 1, 2]
        let words = stride(from: 0, to: samples.count, by: 2).map { i in
            UInt32(Binary16.encode(samples[i])) | (UInt32(Binary16.encode(samples[i + 1])) << 16)
        }
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: words),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 256), .read(buffer: read, byteCount: 256),
        ])
        let actual = try #require(try await ticket.wait().first).map { Float(bitPattern: $0) }
        let exact: [Float] = [
            -2, -2, 2, 2, -1, -1, 1, 1, 0, 0, 1, 1, 0.5, 0.5, 0.5, 0.5,
            0, 3, 15, 17, 0.125, 0.5, 2, 2.25, 0.125, 1, 6, 9, 0, 1, 0, 0.31640625,
            0, 0.31640625, 1.5, 0.5, 3, 0.5, 3, 4,
        ]
        let approximate: [Float] = [
            0.47942554, 0.87758256, 2.35619449, 1.44363548, 1.31695790, 0.54930614, 45, 3.14159265,
            1.64872127, 0.69314718, 3, -1, 0.54630249, 0.52359878, 1.04719755, 0.78539816,
            0.52109531, 1.12762597, 0.46211716, 2.5, -2, -3, -1, 0.5,
        ]
        #expect(actual.count == exact.count + approximate.count)
        #expect(Array(actual.prefix(exact.count)) == exact)
        for (value, expected) in zip(actual.dropFirst(exact.count), approximate) {
            #expect(abs(value - expected) <= 0.004 * max(1, abs(expected)))
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
