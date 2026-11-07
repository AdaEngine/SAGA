import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let selectShader = """
    @group(0) @binding(0) var<storage, read_write> trace: u32;
    @group(0) @binding(1) var<storage, read_write> output: array<u32, 8>;
    fn mark(n: u32) -> u32 { trace = trace * 10u + n; return n; }
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        let scalar = select(mark(2u), mark(3u), true);
        let mask = vec3u(id.x, id.x + 1u, id.x + 2u) < vec3u(1u, 1u, 3u);
        let lanes = select(vec3u(10u, 20u, 30u), vec3u(40u, 50u, 60u), mask);
        let pair = select(vec2i(-1i, -2i), vec2i(3i, 4i), false);
        let floats = select(vec2f(1.5f, 2.5f), vec2f(3.5f, 4.5f), vec2<bool>(false, true));
        let valid = select(false, true, floats.x == 1.5f && floats.y == 4.5f);
        output[0] = scalar;
        output[1] = trace;
        output[2] = lanes.x;
        output[3] = lanes.y;
        output[4] = lanes.z;
        output[5] = u32(pair.x);
        output[6] = u32(pair.y);
        output[7] = select(0u, 1, valid);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func lowersSelectWithScalarAndVectorConditions(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(selectShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    if format == .wgsl {
        #expect(source.contains("select("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    } else {
        #expect(source.contains(" ? "))
    }
}

@Test(arguments: [
    "select(1u, 2i, true)",
    "select(1u, 2u, 1u)",
    "select(vec2u(1u), vec2u(2u), vec3<bool>(true))",
    "select(1u, 2u)",
])
func rejectsInvalidSelectOverloads(expression: String) {
    let source = "@compute @workgroup_size(1) fn run() { _ = \(expression); }"
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func shaderSelectEvaluatesBothOperandsAndChoosesLanesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(selectShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let trace = try await device.createBuffer(.init(size: 4, usage: .storage))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): trace, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 32),
            .read(buffer: read, byteCount: 32),
        ])
        #expect(try await ticket.wait() == [[3, 23, 40, 20, 60, .max, .max - 1, 1]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
