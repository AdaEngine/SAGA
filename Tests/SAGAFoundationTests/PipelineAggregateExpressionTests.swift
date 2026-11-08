import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let pipelineAggregateShader = """
    override scale = 2.0f;
    override lane = 0u;
    override divisor = 2u;
    override width = u32(dot(vec2f(scale, 1.0f), vec2f(1.0f)));
    @group(0) @binding(0) var<storage, read_write> output: array<vec4f, 2>;
    @compute @workgroup_size(width)
    fn run(@builtin(local_invocation_index) i: u32) {
        if i != 0u { return; }
        let lanes = quantizeToF16(vec2f(scale, scale + 1.0f));
        let matrix = transpose(mat2x2f(scale, 1, 2, 3));
        let pair = matrix * vec2f(2, 1);
        let picked = vec4u(11, 22, 33, 44).wzyx[lane + 1u];
        let quotient = vec2u(8, 9) / vec2u(divisor, 3);
        output[0] = vec4f(lanes, pair);
        output[1] = vec4f(f32(picked), f32(quotient.x), f32(quotient.y), f32(width));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func specializesVectorAndMatrixPipelineExpressions(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(pipelineAggregateShader)
    let defaults = try compiler.compileCompute(source, to: format)
    let supplied = try compiler.compileCompute(source, to: format, constants: ["scale": 3, "lane": 1, "divisor": 4])
    #expect(defaults.workgroupSize == .init(x: 3))
    #expect(supplied.workgroupSize == .init(x: 4))
    guard case .text(_, let text) = supplied.artifact else { Issue.record("Expected source"); return }
    #expect(!text.contains("dot(") && !text.contains("transpose(") && !text.contains("quantizeToF16("))
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

@Test(arguments: [
    ("_ = vec2u(4) / vec2u(divisor, 1);", ["divisor": 0.0]),
    ("_ = id.xy / vec2u(divisor, 1);", ["divisor": 0.0]),
    ("_ = id.xy << vec2u(lane, 0);", ["lane": 32.0]),
    ("_ = ldexp(vec2f(id.xy), vec2i(power, 0));", ["power": 129.0]),
    ("_ = quantizeToF16(vec2f(scale, scale + 1.0));", ["scale": 65504.0]),
    ("_ = smoothstep(vec2f(scale, 0), vec2f(1), vec2f(id.xy));", ["scale": 1.0]),
    ("_ = vec2f(scale)[lane + 1u];", ["lane": 1.0]),
    ("_ = mat2x2f(scale, 1, 2, 3)[lane + 1u];", ["lane": 1.0]),
])
func rejectsInvalidAggregatePipelineExpressions(body: String, constants: [String: Double]) throws {
    let source = ShaderInput.wgsl("""
        override divisor = 2u;
        override lane = 0u;
        override power = 0i;
        override scale = 0.5f;
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) { \(body) }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(source)
    _ = try compiler.compileCompute(source, to: .wgsl)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, to: .wgsl, constants: constants)
    }
}

@Test func aggregateLetRemainsRuntimeAfterItsInitializerIsSpecialized() throws {
    let source = ShaderInput.wgsl("""
        override divisor = 2u;
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
            let value = vec2u(divisor, 1);
            _ = id.xy / value;
        }
        """)
    _ = try IndependentShaderCompiler().compileCompute(source, to: .wgsl, constants: ["divisor": 0])
}

@Test func suppliedOverrideSkipsItsAggregateDefault() throws {
    let source = ShaderInput.wgsl("""
        override lane = 0u;
        override width = vec2u(2, 3)[lane];
        @compute @workgroup_size(width) fn run() {}
        """)
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileCompute(source, to: .wgsl, constants: ["width": 5, "lane": 2])
    #expect(artifact.workgroupSize == .init(x: 5))
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, to: .wgsl, constants: ["lane": 2])
    }
}

#if canImport(Metal)
@Test func evaluatesVectorAndMatrixPipelineExpressionsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(pipelineAggregateShader), to: .msl,
                                                                 constants: ["scale": 3, "lane": 1, "divisor": 4])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        ])
        let expected: [Float] = [3, 4, 7, 7, 22, 2, 3, 4]
        #expect(try await ticket.wait() == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
