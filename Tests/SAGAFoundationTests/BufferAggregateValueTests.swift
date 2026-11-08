import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let aggregateBufferShader = """
    struct Record {
        @size(16) tag: u32,
        points: array<vec3u, 2>,
        @size(16) marker: u32,
    }
    struct Config { bias: u32, matrix: mat2x2f, }
    @group(0) @binding(0) var<storage, read> source: Record;
    @group(0) @binding(1) var<storage, read_write> destination: Record;
    @group(0) @binding(2) var<storage, read_write> words: array<u32, 4>;
    @group(0) @binding(3) var<uniform> config: Config;

    fn readRecord() -> Record { return source; }

    @compute @workgroup_size(1) fn run() {
        var snapshot = readRecord();
        let settings = config;
        let previous = words;
        words = array<u32, 4>(previous[3], snapshot.tag, u32(settings.matrix[1].y), settings.bias);
        snapshot.tag += 4u;
        snapshot.points[0].y += settings.bias;
        destination = snapshot;
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func bufferAggregateValuesCompile(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(aggregateBufferShader), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64, 64, 16])
    #expect(shader.uniformBindings.map(\.minimumByteSize) == [24])
    guard case .text(let actual, let code) = shader.artifact else {
        Issue.record("Expected text shader")
        return
    }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code)) }
}

@Test func bufferAggregateValuesUseTheSameBlockInLinkedGLSLStages() throws {
    let source = ShaderInput.wgsl("""
        struct Record { color: vec4f, }
        @group(0) @binding(0) var<storage, read> data: Record;
        @vertex fn vertexMain() -> @builtin(position) vec4f { return data.color; }
        @fragment fn fragmentMain() -> @location(0) vec4f {
            let value = data;
            return value.color;
        }
        """)
    let compiler = IndependentShaderCompiler()
    let blocks = try ["vertexMain", "fragmentMain"].map { entry in
        let shader = try compiler.compileRenderStage(source, entryPoint: entry, to: .glsl)
        guard case .text(.glsl, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
        return try #require(code.split(separator: "\n").first { $0.contains("buffer SagaBuffer0") })
    }
    #expect(blocks[0] == blocks[1])
}

#if canImport(Metal)
@Test func bufferAggregateValuesCopyWithoutWritingPaddingOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(aggregateBufferShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
        let destination = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination, .copySource]))
        let words = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination, .copySource]))
        let config = try await device.createBuffer(.init(size: 24, usage: [.uniform, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let input: [UInt32] = [11, 1001, 1002, 1003, 1, 2, 3, 1004, 4, 5, 6, 1005, 17, 1006, 1007, 1008]
        let matrix = [Float(10), 20, 30, 40].map(\.bitPattern)
        let snapshots = try await device.submitCommands([
            .write(buffer: source, offset: 0, words: input),
            .write(buffer: destination, offset: 0, words: Array(repeating: 900, count: 16)),
            .write(buffer: words, offset: 0, words: [21, 22, 23, 24]),
            .write(buffer: config, offset: 0, words: [7, 999] + matrix),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0): .init(buffer: source), .init(binding: 1): .init(buffer: destination),
                .init(binding: 2): .init(buffer: words), .init(binding: 3): .init(buffer: config),
            ], workgroups: .init(x: 1)),
            .copy(source: destination, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
            .copy(source: words, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [[15, 900, 900, 900, 1, 9, 3, 900, 4, 5, 6, 900, 17, 900, 900, 900], [24, 11, 40, 7]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
