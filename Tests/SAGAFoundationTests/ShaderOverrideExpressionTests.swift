import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let overrideExpressionShader = """
    @id(21) override lanes: u32 = 2u;
    override divisor = 2u;
    const numerator = 8u;
    override bias = numerator / divisor + lanes;
    override enableOffsets = lanes > 1u && (10u / divisor > 2u);
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
    @compute @workgroup_size(min(lanes + 1u, 4u), 1u, 1u)
    fn run(@builtin(local_invocation_index) i: u32) {
        output[i] = select(bias, bias + i, enableOffsets);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func specializesComputedOverrideDefaultsAndWorkgroupSizes(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(overrideExpressionShader)
    let unspecialized = try compiler.parseAndValidate(source).module
    #expect(unspecialized.entryPoints.first?.workgroupOverrides?.x != nil)
    let bias = try #require(unspecialized.overrides.values.first { $0.name == "bias" })
    let initializer = try #require(bias.initializer)
    guard case .binary(op: .add, _, _) = try unspecialized.globalExpressions.value(for: initializer).kind else {
        Issue.record("Expected a typed computed initializer")
        return
    }
    let first = try compiler.compileCompute(source, to: format)
    let second = try compiler.compileCompute(source, to: format, constants: ["21": 1, "divisor": 4])
    #expect(first.workgroupSize == .init(x: 3))
    #expect(second.workgroupSize == .init(x: 2))
    if format == .wgsl {
        guard case .text(_, let specialized) = second.artifact else { Issue.record("Expected source"); return }
        _ = try compiler.parseAndValidate(.wgsl(specialized))
    }
}

@Test func evaluatesOverrideErrorsOnlyForSelectedDefaults() throws {
    let source = ShaderInput.wgsl("""
        override denominator = 0u;
        override calculated = 8u / denominator;
        override width = calculated + 1u;
        @compute @workgroup_size(width) fn run() {}
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(source)
    #expect(try compiler.compileCompute(source, to: .wgsl, constants: ["calculated": 2]).workgroupSize == .init(x: 3))
    #expect(try compiler.compileCompute(source, to: .wgsl, constants: ["denominator": 4]).workgroupSize == .init(x: 3))
    for constants: [String: Double] in [[:], ["width": 3]] {
        do {
            _ = try compiler.compileCompute(source, to: .wgsl, constants: constants)
            Issue.record("Pipeline evaluation accepted an integer division by zero in a statically accessed default")
        } catch ShaderFailure.diagnostics(let issues) {
            #expect(issues.first?.code == .invalidType)
        }
    }
}

@Test func keepsShortCircuitDuringOverrideEvaluation() throws {
    let source = ShaderInput.wgsl("""
        override enabled = false;
        override z = 0.0f;
        override condition = enabled && (1.0f / z > 0.0f);
        override width = select(1u, 2u, condition);
        @compute @workgroup_size(width) fn run() {}
        """)
    let compiler = IndependentShaderCompiler()
    #expect(try compiler.compileCompute(source, to: .wgsl).workgroupSize == .init(x: 1))
    do {
        _ = try compiler.compileCompute(source, to: .wgsl, constants: ["enabled": 1])
        Issue.record("Pipeline evaluation accepted a non-finite result")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .invalidType)
    }
}

@Test func rejectsStaticIntegerDivisionByZeroInOverrideDefault() throws {
    let source = ShaderInput.wgsl("override a = 2u; override b = a / 0u;")
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source)
        Issue.record("Shader creation accepted a constant zero divisor")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .invalidType)
    }
}

@Test func evaluatesTypedCastsAndBitOperationsInOverrides() throws {
    let source = ShaderInput.wgsl("""
        override scale: f32 = 1.25;
        override bits = bitcast<u32>(scale);
        override width = u32(abs(i32(scale * 2.0))) + (bits >> 31u);
        @compute @workgroup_size(width) fn run() {}
        """)
    let compiler = IndependentShaderCompiler()
    #expect(try compiler.compileCompute(source, to: .wgsl).workgroupSize == .init(x: 2))
    #expect(try compiler.compileCompute(source, to: .wgsl, constants: ["scale": -1.75]).workgroupSize == .init(x: 4))
}

#if canImport(Metal)
@Test func evaluatesComputedOverridesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(overrideExpressionShader)
    let first = try compiler.compileCompute(source, to: .msl)
    let second = try compiler.compileCompute(source, to: .msl, constants: ["21": 1, "divisor": 4])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let a = try await device.createComputePipeline(first)
        let b = try await device.createComputePipeline(second)
        let outputA = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let outputB = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: a, bindings: [.init(binding: 0): outputA], workgroups: .init(x: 1)),
            .dispatchCompute(pipeline: b, bindings: [.init(binding: 0): outputB], workgroups: .init(x: 1)),
            .copy(source: outputA, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
            .copy(source: outputB, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[6, 7, 8, 0], [3, 3, 0, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
