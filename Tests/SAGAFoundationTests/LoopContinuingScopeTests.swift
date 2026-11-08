import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import SAGAValidation

private let continuingScopeShader = """
    @group(0) @binding(0) var<storage, read_write> output: u32;
    var<private> calls: u32;

    fn snapshot(step: u32) -> u32 {
        calls++;
        return step + 3u;
    }

    @compute @workgroup_size(1) fn run() {
        var step = 0u;
        var bias = 7u;
        var total = 0u;
        loop {
            let old = bias;
            let fromCall = snapshot(step);
            let pair = vec2u(old, fromCall + step);
            bias += 10u;
            switch step {
                case 0u: { continue; }
                default: {}
            }
            total += 100u;
            continuing {
                total += pair.x + pair.y;
                step++;
                break if fromCall >= 5u;
            }
        }
        output = total * 10u + calls;
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func continuingScopePreservesBodyExpressions(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(continuingScopeShader), to: format)
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    #expect(source.contains("saga_loop_value_"))
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(source))
    } else if format == .glsl {
        _ = try compiler.parseAndValidate(.glsl(source, stage: .compute, defines: [:]))
    }
}

@Test(arguments: [true, false])
func continuingScopeChecksEveryContinueEdge(_ emitBeforeContinue: Bool) throws {
    let condition = try IRHandle<IRExpression>(index: 0)
    let snapshot = try IRHandle<IRExpression>(index: 1)
    let bool = try IRHandle<IRType>(index: 0)
    let blocks = try (0..<5).map { try IRHandle<IRBlock>(index: $0) }
    let statements = try (0..<5).map { try IRHandle<IRStatement>(index: $0) }
    let function = IRFunction(
        expressions: IRTable([
            .init(type: bool, kind: .literal(.bool(true))),
            .init(type: bool, kind: .runtimeValue(condition)),
        ]),
        statements: IRTable([
            .init(kind: .loop(body: blocks[1], continuing: blocks[2], breakIf: snapshot)),
            .init(kind: .emit([snapshot])),
            .init(kind: .ifCondition(condition: condition, accept: blocks[3], reject: blocks[4])),
            .init(kind: .continue),
            .init(kind: .returnValue(nil)),
        ]),
        blocks: IRTable([
            .init(statements: [statements[0], statements[4]]),
            .init(statements: emitBeforeContinue ? [statements[1], statements[2]] : [statements[2], statements[1]]),
            .init(),
            .init(statements: [statements[3]]),
            .init(),
        ]),
        body: blocks[0])
    let module = IRModule(types: IRTable([.init(kind: .scalar(.bool))]), functions: IRTable([function]))
    if emitBeforeContinue {
        try ShaderSemanticValidator().validate(module)
    } else {
        do {
            try ShaderSemanticValidator().validate(module)
            Issue.record("Accepted a continuing operand before its emit")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.contains { $0.code == .invalidReference })
        }
    }
}

@Test func continuingScopeTracksUniformityOfBodyExpressions() throws {
    let source = """
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            var step = 0u;
            loop {
                let test = group.x == 0u;
                continuing {
                    if test { workgroupBarrier(); }
                    step++;
                    break if step == 2u;
                }
            }
        }
        """
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(.wgsl(source))
    do {
        _ = try compiler.parseAndValidate(.wgsl(source.replacing("group.x == 0u", with: "lane == 0u")))
        Issue.record("Accepted a divergent barrier")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .nonUniformControlFlow })
    }
}

#if canImport(Metal)
@Test func continuingScopePreservesSnapshotsAcrossNativeContinueOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(continuingScopeShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 4),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[2663]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
