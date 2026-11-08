import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let halfBufferShader = """
    enable f16;
    struct Input {
        low: f16,
        high: f16,
        @align(8) values: array<vec3h>,
    }
    struct Output {
        @align(4) @size(4) mark: f16,
        values: array<f16>,
    }
    struct Settings {
        low: f16,
        bias: f16,
        tag: u32,
        lanes: vec3h,
        tail: f16,
    }
    @group(0) @binding(0) var<storage, read> input: Input;
    @group(0) @binding(1) var<storage, read_write> output: Output;
    @group(0) @binding(2) var<storage, read_write> vectors: array<vec3h>;
    @group(0) @binding(3) var<uniform> settings: Settings;
    fn sample(i: u32) -> vec3h { return input.values[i]; }
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        let v = sample(id.x);
        vectors[id.x] = v;
        output.values[id.x] = v.x + settings.bias;
        if id.x == 0u {
            output.mark = f16(arrayLength(&input.values) + arrayLength(&output.values));
            vectors[0u].z = settings.tail;
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func preservesHalfPrecisionBufferLayout(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(halfBufferShader), to: format)
    #expect(shader.requiresShaderF16)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [16, 8, 8])
    #expect(shader.storageBindings.map(\.runtimeArrayOffset) == [8, 4, 0])
    #expect(shader.storageBindings.map(\.runtimeArrayStride) == [8, 2, 8])
    #expect(shader.uniformBindings.map(\.minimumByteSize) == [16])
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
    case .msl:
        #expect(source.contains("device half*"))
        #expect(source.contains("constant half*"))
    case .hlsl:
        #expect(shader.storageBindings.map(\.hlslUsesByteAddressBuffer) == [true, true, true])
        #expect(source.contains(".Load<half>(") && source.contains(".Store<half>("))
        #expect(source.contains("count - 4u") && source.contains("/ 2u"))
    case .glsl:
        #expect(source.contains("GL_EXT_shader_16bit_storage"))
        #expect(source.contains("layout(offset = 2) float16_t"))
    default: Issue.record("Unexpected shader format")
    }
}

private func halfPair(_ low: Float, _ high: Float) -> UInt32 {
    UInt32(Binary16.encode(Double(low))) | (UInt32(Binary16.encode(Double(high))) << 16)
}

#if canImport(Metal)
@Test func updatesAdjacentHalfPrecisionBufferElementsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfBufferShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination, .copySource]))
        let vectors = try await device.createBuffer(.init(size: 24, usage: [.storage, .copyDestination, .copySource]))
        let settings = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let readScalars = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
        let readVectors = try await device.createBuffer(.init(size: 24, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [
                halfPair(50, 60), 0, halfPair(1, 2), halfPair(3, -9), halfPair(4, 5), halfPair(6, -9),
                halfPair(7, 8), halfPair(9, -9),
            ]),
            .write(buffer: output, offset: 0, words: [halfPair(99, -7), 0, 0, 0]),
            .write(buffer: vectors, offset: 0, words: Array(repeating: [0, halfPair(0, -7)], count: 3).flatMap { $0 }),
            .write(buffer: settings, offset: 0, words: [halfPair(-5, 0.5), 0, halfPair(11, 12), halfPair(13, 2.5)]),
            .dispatchCompute(pipeline: pipeline, bindings: [
                .init(binding: 0): input, .init(binding: 1): output, .init(binding: 2): vectors, .init(binding: 3): settings,
            ], workgroups: .init(x: 8)),
            .copy(source: output, destination: readScalars, byteCount: 16), .read(buffer: readScalars, byteCount: 16),
            .copy(source: vectors, destination: readVectors, byteCount: 24), .read(buffer: readVectors, byteCount: 24),
        ])
        let actual = try await ticket.wait()
        #expect(actual == [
            [halfPair(9, -7), halfPair(1.5, 4.5), halfPair(7.5, 0.5), halfPair(0.5, 0.5)],
            [halfPair(1, 2), halfPair(2.5, -7), halfPair(4, 5), halfPair(6, -7), halfPair(7, 8), halfPair(9, -7)],
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
