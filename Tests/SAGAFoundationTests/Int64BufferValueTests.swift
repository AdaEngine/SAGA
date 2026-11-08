import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let int64BufferValueShader = """
    struct Params { count: u32, bias: i64, increments: vec3<u64>, }
    struct Cell { signed_value: vec2<i64>, unsigned_value: vec3<u64>, tag: u32, }
    struct Output { marker: u32, cells: array<Cell>, }
    @group(0) @binding(0) var<uniform> params: Params;
    @group(0) @binding(1) var<storage> input: array<Cell>;
    @group(0) @binding(2) var<storage, read_write> output: Output;
    @group(0) @binding(3) var<storage, read_write> scalar_output: array<i64>;

    @compute @workgroup_size(2) fn main(@builtin(local_invocation_index) lane: u32) {
        let cell = input[lane];
        output.cells[lane].signed_value = cell.signed_value + vec2<i64>(params.bias);
        output.cells[lane].unsigned_value = cell.unsigned_value + params.increments;
        output.cells[lane].tag = cell.tag;
        scalar_output[lane] = (cell.signed_value.x * 3li) - 1li;
        if lane == 0u && cell.signed_value.y < params.bias {
            output.marker = params.count + arrayLength(&output.cells);
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func int64BufferValuesKeepLayoutAndArithmetic(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(int64BufferValueShader), to: format)
    #expect(shader.requiresShaderInt64)
    #expect(shader.uniformBindings.map(\.minimumByteSize) == [64])
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64, 96, 8])
    #expect(shader.storageBindings.map(\.runtimeArrayOffset) == [0, 32, 0])
    #expect(shader.storageBindings.map(\.runtimeArrayStride) == [64, 64, 8])
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    switch format {
    case .wgsl:
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.shaderInt64))
    case .msl:
        #expect(source.contains("as_type<long2>"))
        #expect(source.contains("constant ulong*"))
    case .hlsl:
        #expect(shader.hlslProfile == "cs_6_0")
        #expect(source.contains(".Load("))
        #expect(source.contains(".Store2("))
    case .glsl:
        #expect(source.contains("GL_ARB_gpu_shader_int64"))
        #expect(source.contains("u64vec3"))
    default: Issue.record("Unexpected shader target")
    }
}

@Test func int64BufferValuesRequireNativeShaderTarget() {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileWebGPUCompute(.wgsl(int64BufferValueShader))
    }
}

#if canImport(Metal)
@Test func int64BufferValuesRoundTripAndWrapOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(int64BufferValueShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let params = try await device.createBuffer(.init(size: 64, usage: [.uniform, .copyDestination]))
    let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copyDestination, .copySource]))
    let scalar = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 96, usage: [.mapRead, .copyDestination]))
    let padding: UInt32 = 0xdead_beef
    let inputWords: [UInt32] = [
        .max, 0x7fff_ffff, 0, 0x8000_0000, padding, padding, padding, padding,
        .max, .max, 7, 0x8000_0000, .max, 0, 99, padding,
    ]
    let paramWords: [UInt32] = [
        7, padding, 1, 0, padding, padding, padding, padding,
        1, 0, 0, 0x8000_0000, 1, 1, padding, padding,
    ]
    let words = try await device.submitCommands([
        .write(buffer: params, offset: 0, words: paramWords),
        .write(buffer: input, offset: 0, words: inputWords),
        .write(buffer: output, offset: 0, words: Array(repeating: padding, count: 24)),
        .dispatchCompute(pipeline: pipeline,
            bindings: [.init(binding: 0): params, .init(binding: 1): input, .init(binding: 2): output,
                       .init(binding: 3): scalar], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 96), .read(buffer: readback, byteCount: 96),
        .copy(source: scalar, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
    ]).wait()
    #expect(words == [
        [8, padding, padding, padding, padding, padding, padding, padding,
         0, 0x8000_0000, 1, 0x8000_0000, padding, padding, padding, padding,
         0, 0, 7, 0, 0, 2, 99, padding],
        [0xffff_fffc, 0x7fff_ffff],
    ])
    await device.close()
}
#endif
