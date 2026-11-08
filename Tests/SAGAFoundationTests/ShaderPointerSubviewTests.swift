import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let pointerSubviewShader = """
    struct Packet { samples: array<u32, 2>, sentinel: u32, }
    override base = 30u;
    var<private> state = Packet(array(base, 40u), 222u);
    var<private> calls: u32;
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 4>;

    fn modify(p: ptr<function, u32>, q: ptr<private, u32>) -> vec2u {
        calls += 1u;
        let previous = *p + *q;
        *p += 3u;
        *q += 7u;
        return vec2u(previous, *p + *q);
    }

    fn forward(p: ptr<function, u32>, q: ptr<private, u32>) -> vec2u {
        return modify(&*p, &*q);
    }

    fn touch(unused: ptr<private, u32>, lane: u32) {
        state.samples[lane] += 1u;
    }

    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32) {
        var packet = Packet(array(10u, 20u), 111u);
        var index = i32(lane) - 1i;
        let packetRef = &packet;
        let samples = &packetRef.samples;
        let p = &samples[index];
        let stateRef = &state;
        let q = &stateRef.samples[lane];
        index = 0i;
        let result = forward(p, q);
        touch(q, lane);
        output[lane] = vec4u(result, packet.samples[0] + packet.samples[1] + packet.sentinel + 1000u * calls,
                            state.samples[0] + state.samples[1] + state.sentinel);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func compilesShaderPointerSubviewArguments(format: ShaderOutputFormat, explicitRequirements: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let requirements = explicitRequirements ? "requires unrestricted_pointer_parameters, pointer_composite_access;\n" : ""
    let shader = try compiler.compileCompute(.wgsl(requirements + pointerSubviewShader), to: format, constants: ["base": 50])
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl {
        #expect(!generated.contains("requires unrestricted_pointer_parameters"))
        _ = try compiler.parseAndValidate(.wgsl(generated))
    }
}

@Test func rejectsWriteAliasingBetweenShaderPointerSubviews() throws {
    let source = ShaderInput.wgsl("""
        fn modify(a: ptr<function, u32>, b: ptr<function, u32>) { *a = *b; }
        @compute @workgroup_size(1) fn run() {
            var values = array(1u, 2u);
            modify(&values[0], &values[1]);
        }
        """)
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source)
        Issue.record("Accepted aliased subviews")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .invalidReference && $0.message.contains("Aliased pointer arguments") })
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesWholeStoragePointerWithoutRequires(format: ShaderOutputFormat) throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> value: u32;
        fn increase(p: ptr<storage, u32, read_write>) { *p += 1u; }
        @compute @workgroup_size(1) fn run() { increase(&value); }
        """)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(source, to: format)
    if case .text(.wgsl, let code) = shader.artifact { _ = try compiler.parseAndValidate(.wgsl(code)) }
}

@Test(arguments: [ShaderOutputFormat.hlsl, .glsl])
func readOnlyShaderPointerParametersAvoidCopyBack(format: ShaderOutputFormat) throws {
    let source = ShaderInput.wgsl("""
        var<private> counter: u32 = 1u;
        fn touch(unused: ptr<private, u32>) { counter = 7u; }
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(1) fn run() { touch(&counter); output = counter; }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: format)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    #expect(generated.contains("in uint saga_param_0"))
}

#if canImport(Metal)
@Test func shaderPointerSubviewsPreserveRobustnessOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(pointerSubviewShader), to: .msl,
                                                                 constants: ["base": 50])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        ])
        #expect(try await ticket.wait() == [[
            50, 57, 1141, 320, 50, 60, 1144, 320, 20, 23, 1144, 312, 0, 0, 1141, 312,
        ]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
