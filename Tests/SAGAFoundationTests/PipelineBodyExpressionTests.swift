import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let pipelineBodyShader = """
    override divisor = 2u;
    override lane = 0u;
    @group(0) @binding(0) var<storage, read> input: array<u32, 4>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32, 4>;
    fn unused() -> u32 { return 4u / (divisor - divisor); }
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        output[0] = input[lane + 1u] / divisor;
        output[1] = (12u / divisor) + id.x;
        let runtimeIndex = lane + 1u;
        output[2] = input[runtimeIndex];
    }
    """

@Test(arguments: [
    ("output[0] = id.x / divisor;", ["divisor": 0.0]),
    ("output[0] = 8u / (divisor - 1u);", ["divisor": 1.0]),
    ("output[0] = input[lane + 1u];", ["lane": 3.0]),
    ("output[0] = bitcast<u32>(quantizeToF16(scale + 1.0));", ["scale": 65504.0]),
    ("output[0] = bitcast<u32>(ldexp(f32(id.x), power + 1i));", ["power": 128.0]),
    ("output[0] = bitcast<u32>(smoothstep(scale, 1.0f, f32(id.x)));", ["scale": 1.0]),
    ("output[0] = extractBits(id.x, lane, 2u);", ["lane": 31.0]),
    ("output[0] = id.x << lane;", ["lane": 32.0]),
])
func rejectsInvalidPipelineExpressionsInFunctionBodies(body: String, constants: [String: Double]) throws {
    let source = ShaderInput.wgsl("""
        override divisor = 2u;
        override lane = 0u;
        override scale = 0.0f;
        override power = 0i;
        @group(0) @binding(0) var<storage, read> input: array<u32, 4>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 4>;
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) { \(body) }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(source)
    _ = try compiler.compileCompute(source, to: .wgsl)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, to: .wgsl, constants: constants)
    }
}

@Test func letKeepsSpecializedOperandsAtRuntime() throws {
    let source = ShaderInput.wgsl("""
        override divisor = 2u;
        override lane = 0u;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
            let index = lane + 1u;
            let denominator = divisor;
            output[index] = id.x / denominator;
        }
        """)
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileCompute(source, to: .wgsl, constants: ["lane": 3, "divisor": 0])
    guard case .text(_, let generated) = artifact.artifact else { Issue.record("Expected shader source"); return }
    _ = try compiler.parseAndValidate(.wgsl(generated))
}

@Test func evaluatesOnlyNeededOverrideShortCircuitSubexpressionsInBody() throws {
    let source = ShaderInput.wgsl("""
        override enabled = false;
        override z = 0.0f;
        @compute @workgroup_size(1) fn run() {
            let flag = enabled && 1.0f / z > 0.0f;
        }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileCompute(source, to: .wgsl)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, to: .wgsl, constants: ["enabled": 1])
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func specializesScalarPipelineExpressionsInBody(format: ShaderOutputFormat) throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(pipelineBodyShader), to: format,
                                                                 constants: ["divisor": 4, "lane": 1])
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(!source.contains("override "))
    if format == .wgsl { _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func evaluatesScalarPipelineExpressionsInBodyOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(pipelineBodyShader), to: .msl,
                                                                 constants: ["divisor": 4, "lane": 1])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [6, 8, 10, 12]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[2, 3, 10, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

@Test func letBoundOverrideKeepsUniformControlFlow() throws {
    let source = ShaderInput.wgsl("""
        override enabled = true;
        @compute @workgroup_size(2) fn run() {
            let flag = enabled;
            if flag { workgroupBarrier(); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(source)
    let shader = try compiler.compileCompute(source, to: .wgsl)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected shader source"); return }
    _ = try compiler.parseAndValidate(.wgsl(generated))
}
