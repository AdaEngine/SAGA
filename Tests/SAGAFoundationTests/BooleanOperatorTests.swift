import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let eagerBooleanShader = """
    const_assert all((vec2(true, false) & vec2(true, true)) == vec2(true, false));
    const_assert all(!(vec2(false, true) | vec2(false, false)) == vec2(true, false));

    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 6>;
    var<private> trace: u32;

    fn mark(tag: u32, value: bool) -> bool {
        trace = trace * 10u + tag;
        return value;
    }

    fn markVector(tag: u32, value: vec4<bool>) -> vec4<bool> {
        trace = trace * 10u + tag;
        return value;
    }

    @compute @workgroup_size(1) fn run() {
        let both = mark(1u, false) & mark(2u, true);
        let either = mark(3u, true) | mark(4u, false);
        let short = (false && mark(9u, true)) || (true || mark(9u, false));
        output[0] = vec4u(u32(both), u32(either), trace, u32(short));
        let lhs = vec4(true, true, false, false);
        let rhs = vec4(true, false, true, false);
        output[1] = select(vec4u(0), vec4u(1), markVector(5u, lhs) & markVector(6u, rhs));
        output[2] = select(vec4u(0), vec4u(1), !(markVector(7u, lhs) | markVector(8u, rhs)));
        output[3] = select(vec4u(0), vec4u(1), lhs == rhs);
        output[4] = select(vec4u(0), vec4u(1), lhs != rhs);
        var combined = lhs;
        combined &= rhs;
        combined |= vec4(false, false, false, true);
        output[5] = vec4u(trace, u32(combined.x), u32(combined.y), u32(combined.w));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func booleanOperatorsTranslateEagerScalarAndVectorOverloads(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(eagerBooleanShader), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [96])
    guard case .text(let target, let text) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(target == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

@Test func booleanOperatorsRejectInvalidShapesAndOperators() throws {
    let compiler = IndependentShaderCompiler()
    for expression in ["true ^ false", "true | 1u", "true & vec2(true)",
                       "vec2(false) | vec3(true)", "vec2(false) && vec2(true)", "~vec2(false)",
                       "vec2(false) < vec2(true)"] {
        let shader = "@compute @workgroup_size(1) fn run() { let a = \(expression); _ = a; }"
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(shader)) }
    }
}

#if canImport(Metal)
@Test func booleanOperatorsPreserveSideEffectsAndLaneValuesOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(eagerBooleanShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 96, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 96),
            .read(buffer: readback, byteCount: 96),
        ]).wait()
        #expect(snapshots == [[0, 1, 1234, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 12345678, 1, 0, 1]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
