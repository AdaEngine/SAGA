import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let abstractIndexShader = """
    const lanes = vec3(9, -4, 7);
    const pairs = array(vec2(21, 22), vec2(33, 34));
    const transform = mat3x2(1, 2, 3, 4, 5, 6);
    const large = vec2(2147483648, 7);
    const last = 2u;
    @id(4) override selected = 1u;
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 5>;

    @compute @workgroup_size(5)
    fn run(@builtin(local_invocation_index) lane: u32) {
        let i = i32(lane) - 1i;
        let value: i32 = lanes[i];
        let pair: vec2i = pairs[i];
        let column: vec2f = transform[i];
        let named: u32 = lanes[last];
        let folded: u32 = large[last - 1u];
        output[lane] = vec4u(bitcast<u32>(value), u32(pair.y), u32(column.y),
                            named + folded + u32(lanes[selected]));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func abstractAggregateIndexingConcretizesOnlyForNonConstantIndices(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(abstractIndexShader), to: format, constants: ["4": 0])
    #expect(shader.storageBindings.first?.minimumByteSize == 80)
    guard case .text(let actual, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code)) }
}

@Test(arguments: [
    "const v = vec2(1, 2); fn read(i: u32) -> u32 { return v[i]; }",
    "const a = array(1, 2); fn read(i: u32) -> u32 { return a[i]; }",
    "const m = mat2x2(1, 2, 3, 4); fn read(i: u32) -> vec2h { return m[i]; }",
])
func abstractAggregateIndexingRejectsContextualConversionAfterRuntimeAccess(_ declaration: String) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("enable f16; " + declaration), capabilities: [.shaderFloat16])
        Issue.record("Accepted a contextual conversion after runtime indexing")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

@Test func abstractAggregateIndexingRejectsOutOfRangeConstantAndOverrideIndices() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl("const m = mat2x2(1, 2, 3, 4); fn read() -> vec2f { return m[2u]; }"))
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(.wgsl(abstractIndexShader), to: .msl, constants: ["4": 3])
    }
}

#if canImport(Metal)
@Test func abstractAggregateIndexingReadsRuntimeElementsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(abstractIndexShader), to: .msl, constants: ["4": 0])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 80, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 80), .read(buffer: readback, byteCount: 80),
        ]).wait()
        #expect(snapshots == [[0, 0, 0, 23, 9, 22, 2, 23, 0xfffffffc, 34, 4, 23, 7, 0, 6, 23, 0, 0, 0, 23]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
