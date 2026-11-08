import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let pipelineCompositeShader = """
    struct Pair { count: u32, values: array<u32, 2>, }
    override first = 3u;
    override lane = 0i;
    override width = Pair(2u, array<u32, 2>(first, first + 1u)).values[lane];
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
    @compute @workgroup_size(width)
    fn run(@builtin(local_invocation_index) id: u32) {
        if id >= 2u { return; }
        let pair = Pair(first, array<u32, 2>(width, first + 10u));
        output[id] = pair.values[id] + pair.count;
        output[id + 2u] = array<Pair, 2>(Pair(), Pair(first, array<u32, 2>()))[lane].count;
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func specializesArrayAndStructurePipelineExpressions(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(pipelineCompositeShader)
    let artifact = try compiler.compileCompute(source, to: format, constants: ["first": 4, "lane": 1])
    #expect(artifact.workgroupSize == .init(x: 5))
    guard case .text(_, let generated) = artifact.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
}

@Test(arguments: [-1.0, 2.0])
func rejectsOutOfBoundsArrayPipelineIndex(index: Double) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(pipelineCompositeShader)
    _ = try compiler.parseAndValidate(source)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, to: .wgsl, constants: ["lane": index])
    }
    let runtimeArray = ShaderInput.wgsl("""
        override lane = 0i;
        @compute @workgroup_size(1) fn run(@builtin(local_invocation_index) id: u32) {
            _ = array<u32, 2>(id, id + 1u)[lane];
        }
        """)
    _ = try compiler.parseAndValidate(runtimeArray)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(runtimeArray, to: .wgsl, constants: ["lane": index])
    }
}

#if canImport(Metal)
@Test func evaluatesArrayAndStructurePipelineExpressionsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(pipelineCompositeShader), to: .msl,
                                                                 constants: ["first": 4, "lane": 1])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[9, 18, 4, 4]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
