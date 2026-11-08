import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let pipelineBuiltinShader = """
    override amplitude = 3.5f;
    override active = true;
    override width = select(1u, u32(frexp(amplitude).exp + i32(modf(amplitude).whole)),
                            all(vec2<bool>(active, amplitude > 0.0f)));
    @group(0) @binding(0) var<storage, read_write> output: array<vec4f, 2>;
    @compute @workgroup_size(width)
    fn run(@builtin(local_invocation_index) id: u32) {
        if id != 0u { return; }
        let split = frexp(vec2f(amplitude, -amplitude));
        let parts = modf(vec2f(amplitude, -amplitude));
        let enabled = any(vec2<bool>(false, active));
        output[0] = vec4f(split.fract, parts.fract);
        output[1] = vec4f(f32(split.exp.x), parts.whole.y,
                          select(0.0f, 1.0f, all(all(vec2<bool>(id == 0u, enabled)))),
                          select(0.0f, 1.0f, any(vec2<bool>(id != 0u, false))));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesPipelineDecompositionAndBooleanReductions(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(pipelineBuiltinShader)
    let defaults = try compiler.compileCompute(input, to: format)
    let specialized = try compiler.compileCompute(input, to: format, constants: ["amplitude": 6.75])
    let disabled = try compiler.compileCompute(input, to: format, constants: ["active": 0])
    #expect(defaults.workgroupSize == .init(x: 5))
    #expect(specialized.workgroupSize == .init(x: 9))
    #expect(disabled.workgroupSize == .init(x: 1))
    guard case .text(_, let generated) = specialized.artifact else { Issue.record("Expected source"); return }
    #expect(!generated.contains("frexp(") && !generated.contains("modf("))
    #expect(generated.contains("all(") && generated.contains("any("))
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
}

@Test func specializesHalfPrecisionDecompositionDefaults() throws {
    let source = ShaderInput.wgsl("""
        enable f16;
        override scale = 2.5h;
        override width = u32(frexp(vec2h(scale)).exp.y + i32(modf(scale).whole));
        @compute @workgroup_size(width) fn run() {}
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl, constants: ["scale": 4.75])
    #expect(shader.workgroupSize == .init(x: 7))
}

@Test(arguments: [
    ("_ = all(vec2u(0u, 1u));", ShaderDiagnosticCode.typeMismatch),
    ("_ = any(true, false);", .typeMismatch),
    ("if all(vec2<bool>(id == 0u, true)) { workgroupBarrier(); }", .nonUniformControlFlow),
])
func rejectsInvalidBooleanReduction(body: String, code: ShaderDiagnosticCode) throws {
    let input = ShaderInput.wgsl("""
        @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) id: u32) { \(body) }
        """)
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Invalid reduction or control flow was accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == code })
    }
}

#if canImport(Metal)
@Test func evaluatesPipelineDecompositionAndBooleanReductionsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(pipelineBuiltinShader), to: .msl,
                                                                 constants: ["amplitude": 6.75])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        ])
        let expected: [Float] = [0.84375, -0.84375, 0.75, -0.75, 3, -6, 1, 0]
        #expect(try await ticket.wait() == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
