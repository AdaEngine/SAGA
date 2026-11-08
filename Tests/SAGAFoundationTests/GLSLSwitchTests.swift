import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let glslSwitchShader = ShaderInput.glsl("""
    #version 450
    layout(local_size_x=1) in;
    layout(set=0,binding=0,std430) buffer Data { uint words[8]; } data;
    const uint BASE = 1u;

    uint selectIndex(uint index) {
        data.words[7] += 1u;
        return index;
    }

    uint choose(uint value) {
        switch (value) {
            case 0u: value = 1u;
            case 1u: switch (value) { default: return 11u; }
            default: return 99u;
        }
    }

    void record(uint value) { data.words[2] = choose(value); }

    void main() {
        const uint offset = BASE + 1u;
        uint sum = 0u;
        for (uint i=0u; i<5u; i++) {
            switch (selectIndex(i)) {
                case 0u:
                case 1u: sum += 2u;
                default:
                    sum += 10u;
                    if (i == 0u) { continue; }
                case offset: sum += 20u; break;
                case 3:
                    switch (i) { case 3u: sum += 30u; break; }
                    sum += 3u;
                    break;
            }
            sum += 100u;
        }
        data.words[0] = sum;
        data.words[1] = choose(0);
        record(5);
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslSwitchTranslatesFallthroughAndNestedJumps(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(glslSwitchShader, to: format)
    guard case .text(_, let text) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(text))
    } else if format == .glsl {
        _ = try compiler.parseAndValidate(.glsl(text, stage: .compute, defines: [:]))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslSwitchTranslatesFragmentCasesSharingALexicalScope(_ format: ShaderOutputFormat) throws {
    let input = ShaderInput.glsl("""
        #version 450
        layout(location=0) out vec4 color;
        void main() {
            float channel = 0.0;
            switch (int(gl_FragCoord.x)) {
                case 0: float shade = 0.25;
                case 1: shade = 0.5; channel = shade; break;
                default: channel = 1.0;
            }
            color = vec4(channel, 0.0, 0.0, 1.0);
        }
        """, stage: .fragment, defines: [:])
    let result = try IndependentShaderCompiler().compileRenderStage(input, entryPoint: "main", to: format)
    if case .text(let generated, _) = result.artifact {
        #expect(generated == format)
    } else {
        Issue.record("Expected shader text")
    }
}

@Test(arguments: [
    ("switch (true) {}", ShaderDiagnosticCode.typeMismatch),
    ("switch (1u) { case 1: break; case 1u: break; }", .duplicateName),
    ("switch (1) { default: ; default: ; }", .duplicateName),
    ("switch (1) { case 1: }", .unexpectedToken),
    ("switch (1) { int x = 0; case 1: break; }", .unexpectedToken),
    ("int x = 1; switch (x) { case x: break; }", .invalidType),
    ("switch (1) { case 1: continue; }", .invalidControlFlow),
])
func glslSwitchDiagnosesInvalidSelectorsAndJumps(_ body: String, _ code: ShaderDiagnosticCode) {
    let input = ShaderInput.glsl("#version 450\nvoid main() { \(body) }", stage: .compute, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Accepted invalid GLSL switch")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == code)
    } catch {
        Issue.record("Unexpected diagnostic: \(error)")
    }
}

@Test func glslSwitchJoinsValuesWrittenByFallthroughBeforeBarriers() throws {
    let compiler = IndependentShaderCompiler()
    func input(_ ending: String) -> ShaderInput {
        .glsl("""
            #version 450
            layout(local_size_x=4) in;
            void main() {
                uint flag = 0u;
                switch (0u) {
                    case 0u: flag = gl_LocalInvocationIndex; \(ending)
                    default: if (flag == 0u) { barrier(); }
                }
            }
            """, stage: .compute, defines: [:])
    }
    _ = try compiler.parseAndValidate(input("break;"))
    do {
        _ = try compiler.parseAndValidate(input(""))
        Issue.record("Accepted a barrier reached through a varying assignment")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .nonUniformControlFlow)
    }
}

#if canImport(Metal)
@Test func glslSwitchRunsFallthroughAndLoopJumpsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(glslSwitchShader, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .write(buffer: output, offset: 0, words: Array(repeating: 0, count: 8)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 32),
            .read(buffer: readback, byteCount: 32),
        ]).wait()
        #expect(result == [[527, 11, 99, 0, 0, 0, 0, 5]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
