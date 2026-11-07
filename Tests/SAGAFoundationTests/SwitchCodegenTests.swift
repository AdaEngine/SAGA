import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let switchShader = """
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 8>;
    const alternative = select(-2i, -1i, true);
    fn mapped(i: i32) -> u32 {
        switch i {
            case -2i, alternative: { return 7u; }
            case 0i: { return 11u; }
            case 9i, (-2147483647i - 1i), default: { return 19u; }
        }
    }
    fn choose() -> u32 { output[7] += 1u; return 2u; }
    @compute @workgroup_size(1)
    fn run() {
        var total = 0u;
        var i = 0u;
        loop {
            switch i {
                case 0u, 2u: { total += 2u; continue; }
                case 1u: { total += 10u; break; }
                default: { total += 1u; }
            }
            total += 100u;
            continuing { i += 1u; break if i >= 4u; }
        }
        output[0] = mapped(-2i); output[1] = mapped(-1i); output[2] = mapped(0i);
        output[3] = mapped(9i); output[4] = mapped(5i); output[5] = total;
        switch choose() {
            case 2u: { output[6] = 23u; }
            default: { output[6] = 99u; }
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsSwitchWithGroupedConstantSelectorsAndLoopJumps(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(switchShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(source.contains("switch") && source.contains("continue;"))
    if format == .wgsl {
        #expect(source.contains("case -2i, -1i:"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    } else {
        #expect(source.contains("case -2:") && source.contains("case -1:") && source.contains("default:"))
    }
}

@Test(arguments: [
    "switch 1u { case 1u: {} case 2u - 1u: {} default: {} }",
    "switch 1u { case 1i: {} default: {} }",
    "switch 1u { case 1u, default: {} default: {} }",
    "switch 1u { case 1u: {} }",
    "switch 1u { default: { continue; } }",
])
func rejectsInvalidSwitchSelectorsAndTargets(body: String) {
    let source = "@compute @workgroup_size(1) fn run() { \(body) }"
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

@Test func switchBreakReconvergesButBarriersInsideDivergentCasesFail() throws {
    let prefix = "@compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) "
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(prefix + """
        { switch lane { case 0u: { break; } default: {} } workgroupBarrier(); }
        """))
    for invalid in [
        "{ switch lane { case 0u: { workgroupBarrier(); } default: {} } }",
        "{ switch 0u { case 0u: { { if lane == 0u { break; } } workgroupBarrier(); } default: {} } }",
    ] {
        do {
            _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(prefix + invalid))
            Issue.record("Accepted a divergent barrier")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.contains { $0.code == .nonUniformControlFlow })
        }
    }
}

#if canImport(Metal)
@Test func switchBreakAndContinuePreserveLoopTargetsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(switchShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 32),
            .read(buffer: read, byteCount: 32),
        ])
        #expect(try await ticket.wait() == [[7, 7, 11, 19, 19, 215, 23, 1]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
