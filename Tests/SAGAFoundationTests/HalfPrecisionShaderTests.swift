import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let halfShader = """
    enable f16;
    alias HalfPair = vec2h;
    const half: f16 = 1.00048828125;
    const_assert half == 1.0h;
    const_assert f16(0x1.002p0) == 0x1p0h;
    const_assert f16(2i) == 2h;
    const_assert f32(f16(1.0009765625)) == 1.0009765625f;
    const_assert modf(vec2h(-3.25, 2.75)).whole.x == -3h;
    const_assert frexp(vec3h(4, -0.125, 65504)).exp.z == 16i;
    const_assert ldexp(0.5h, 16) == 32768h;
    const_assert ldexp(0.5h, -23i) == 0x1p-24h;
    const_assert ldexp(0.5h, 1i) + 0x1p-11h == 1h;
    @group(0) @binding(0) var<storage, read> input: array<vec4f, 1>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 3>;
    fn splitPair(x: HalfPair) -> vec4f {
        var parts = modf(x);
        parts.whole = -parts.whole;
        return vec4f(vec2f(parts.fract), vec2f(parts.whole));
    }
    @compute @workgroup_size(1) fn run() {
        let x = vec4h(input[0]);
        let parts = frexp(x);
        let triple = frexp(x.xyz);
        let scalar = frexp(x.y);
        output[0] = vec4f(ldexp(parts.fract, parts.exp));
        output[1] = splitPair(x.yz);
        output[2] = vec4f(f32(triple.fract.x), f32(triple.exp.z), f32(scalar.fract), f32(scalar.exp));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsHalfPrecisionDecomposition(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(halfShader), to: format)
    #expect(shader.requiresShaderF16)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(source.contains("modf(") && source.contains("ldexp("))
    if format == .hlsl {
        #expect(source.contains("firstbithigh(") && !source.contains("frexp("))
    } else {
        #expect(source.contains("frexp("))
    }
    if format == .wgsl {
        #expect(source.hasPrefix("enable f16;"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsHalfPrecisionInFragmentStage(format: ShaderOutputFormat) throws {
    let source = """
        diagnostic(off, derivative_uniformity);
        enable f16;
        @fragment fn color(@location(0) x: f32) -> @location(0) vec4f {
            let parts = modf(f16(x));
            return vec4f(f32(parts.fract), f32(parts.whole), 0, 1);
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(source), entryPoint: "color", to: format)
    #expect(shader.requiresShaderF16)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(body.contains("modf("))
    if format == .wgsl {
        #expect(body.hasPrefix("enable f16;"))
        _ = try compiler.parseAndValidate(.wgsl(body), capabilities: [.shaderFloat16])
    }
    if format == .hlsl { #expect(shader.hlslProfile == "ps_6_2") }
}

@Test(arguments: [
    ("const x: f16 = 1;", ShaderDiagnosticCode.unsupportedCapability),
    ("const x = 1h;", .unsupportedCapability),
    ("const x = vec2h(1);", .unsupportedCapability),
    ("const x = 1; enable f16;", .unsupportedConstruct),
    ("enable f16; const x = 65536.0h;", .invalidToken),
    ("enable f16; const x = 0x1.002p0h;", .invalidToken),
    ("enable f16; const x = 0x1.0h;", .invalidToken),
    ("enable f16; @compute @workgroup_size(1) fn run() { let x = f16(70000.0f); }", .invalidType),
    ("enable f16; const x = 1.0h + 1.0f;", .typeMismatch),
    ("enable f16; const x = ldexp(0.5h, 17);", .invalidType),
    ("enable f16; @compute @workgroup_size(1) fn run() { var x = 0.5h; let y = ldexp(x, 17); }", .invalidType),
])
func diagnosesHalfPrecisionShaderErrors(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
        Issue.record("Invalid half-precision shader accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

@Test func halfPrecisionRequiresCompilerCapability() throws {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl("enable f16; const x = 1h;"))
    }
}

@Test func halfPrecisionPipelineRequiresDeviceFeature() async throws {
    let device = try await Device.make(backend: TestBackend())
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfShader), to: .wgsl)
    await #expect(throws: GPUError.validation("Feature not advertised")) {
        try await device.createComputePipeline(shader)
    }
    await device.close()
}

#if canImport(Metal)
@Test func evaluatesHalfPrecisionDecompositionOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfShader), to: .msl)
    let values: [Float] = [65504, -3.25, 2.75, 0x1p-14]
    let expected = values + [-0.25, 0.75, 3, -2, 0.99951171875, 2, -0.8125, 2]
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: values.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
        ])
        #expect(try await ticket.wait().first == expected.map(\.bitPattern))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
