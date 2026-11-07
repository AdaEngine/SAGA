import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let constantSelectShader = """
    const enabled = select(false, true, true);
    const scale = select(4u, 8u, enabled);
    const bias = select(-2i, -9i, false);
    const factor = select(0.25f, 0.5f, enabled);
    const lanes = select(vec4u(1u, 2u, 3u, 4u), vec4u(8u, 9u, 10u, 11u), vec4<bool>(true, false, true, false));
    const_assert select(0u, scale, enabled) == 8u;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, select(2u, 4u, enabled)>;
    fn adjustment() -> u32 { return scale + u32(bias + 10i); }
    @compute @workgroup_size(select(2, 4, enabled))
    fn run(@builtin(global_invocation_id) id: vec3u) {
        if enabled && factor == 0.5f {
            output[id.x] = adjustment() + lanes.x + lanes.y + lanes.z + lanes.w + id.x;
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsFoldedSelectConstantsInComputeShaders(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(.wgsl(constantSelectShader)).module
    #expect(module.constants.count == 4)
    #expect(module.entryPoints.first?.workgroupSize == .init(x: 4))
    let shader = try compiler.compileCompute(.wgsl(constantSelectShader), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [16])
    #expect(shader.workgroupSize == .init(x: 4))
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected source"); return }
    #expect(!source.contains("select("))
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(source))
    } else if format == .glsl {
        _ = try compiler.parseAndValidate(.glsl(source, stage: .compute, defines: [:]))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsFoldedSelectConstantsInRenderShaders(to format: ShaderOutputFormat) throws {
    let source = """
        const red = select(0.25f, 0.5f, true);
        const rest = select(vec3f(0.25f, 0.5f, 1.0f), vec3f(0.75f, 0.75f, 0.25f), vec3<bool>(true, false, false));
        fn color() -> vec4f { return vec4f(red, rest); }
        @fragment fn run() -> @location(0) vec4f { return color(); }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(source), entryPoint: "run", to: format)
    guard case .text(_, let result) = shader.artifact else { Issue.record("Expected source"); return }
    #expect(result.contains("0.5") && result.contains("0.75") && !result.contains("select("))
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(result)) }
}

@Test func rejectsInvalidConstantSelectOperandEvenWhenUnselected() {
    let source = """
        const invalid = select(1u, 2u / 0u, false);
        @compute @workgroup_size(1) fn run() {}
        """
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func foldedSelectConstantsDetermineLayoutAndValuesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(constantSelectShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 16),
            .read(buffer: read, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[40, 41, 42, 43]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
