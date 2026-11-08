import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let unreachableShader = """
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 2>;

    fn selected(lane: u32) -> u32 {
        switch lane {
            case 0u: {
                return 7u;
                output[0].x = 999u;
                break;
            }
            default: { return 11u; }
        }
        output[1].x = 999u;
    }

    fn loopValue() -> u32 {
        loop {
            return 23u;
            break;
        }
    }

    fn deadBarrier() {
        return;
        workgroupBarrier();
    }

    fn loopCounter() -> u32 {
        var counter = 0u;
        loop {
            counter = 17u;
            break;
            counter = 99u;
            continuing { counter = 123u; }
        }
        return counter;
    }

    @compute @workgroup_size(2)
    fn run(@builtin(local_invocation_index) lane: u32) {
        if lane == 0u { deadBarrier(); }
        output[lane] = vec4u(selected(lane), loopValue(), loopCounter(), 5u);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func unreachableWGSLRetainsLiveReturnAndLoopBehavior(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(unreachableShader), to: format)
    guard case .text(let actual, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code)) }
}

@Test(arguments: [
    ("fn bad() -> u32 { return 1u; return true; }", ShaderDiagnosticCode.typeMismatch),
    ("fn bad() { return; _ = missing; }", .unresolvedName),
    ("fn bad() { return; break; }", .invalidControlFlow),
    ("@compute @workgroup_size(1) fn bad() { return; discard; }", .invalidControlFlow),
    ("fn bad() -> u32 { if true { return 1u; } }", .invalidControlFlow),
])
func unreachableWGSLStillChecksDeadStatements(_ source: String, _ diagnostic: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Accepted an invalid statement after a return")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == diagnostic)
    }
}

@Test func unreachableWGSLDoesNotEmitDuplicateGLSLReturnTemporaries() throws {
    let source = ShaderInput.wgsl("""
        struct Color { @location(0) value: vec4f }
        @fragment fn draw() -> Color {
            return Color(vec4f(1.0));
            return Color(vec4f(2.0));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileRenderStage(source, entryPoint: "draw", to: .glsl).artifact
    guard case .text(.glsl, let text) = artifact else { throw ShaderFailure.invalidHandle }
    #expect(text.split(separator: "\n").filter { $0.contains("SagaFragmentOutput saga_result =") }.count == 1)
}

#if canImport(Metal)
@Test func unreachableWGSLDoesNotExecuteDeadStoresOrContinuingOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(unreachableShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
        ]).wait()
        #expect(snapshots == [[7, 23, 17, 5, 11, 23, 17, 5]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
