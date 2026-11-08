import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let nestedStorageShader = """
    struct Row { values: array<vec3u, 2>, @size(16) marker: u32, }
    struct Data { @size(16) generation: u32, rows: array<Row>, }
    @group(0) @binding(0) var<storage, read_write> data: Data;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;

    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let outer = i32(lane) - 1i;
        let inner = select(1u, 2u, lane == 2u);
        let before = data.rows[outer].values[inner];
        data.rows[outer].values[inner].z = 70u + lane;
        var row = data.rows[outer];
        row.marker += lane;
        data.rows[outer] = row;
        output[lane] = vec4u(before, arrayLength(&data.rows) + data.rows[outer].marker);
    }
    """

private let nestedInput: [UInt32] = [
    7, 700, 701, 702,
    1, 2, 3, 703, 4, 5, 6, 704, 100, 705, 706, 707,
    11, 12, 13, 708, 14, 15, 16, 709, 200, 710, 711, 712,
]

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesNestedStorageBufferLayouts(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(nestedStorageShader), to: format)
    let storage = try #require(shader.storageBindings.first)
    #expect(storage.minimumByteSize == 64)
    #expect(storage.runtimeArrayOffset == 16)
    #expect(storage.runtimeArrayStride == 48)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
}

#if canImport(Metal)
@Test func nestedStorageCopiesRespectBoundsAndPaddingOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(nestedStorageShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let data = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: data, offset: 0, words: nestedInput),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
            .copy(source: data, destination: read, byteCount: 112), .read(buffer: read, byteCount: 112),
        ])
        var updated = nestedInput
        updated[10] = 71
        updated[12] = 101
        updated[24] = 202
        #expect(try await ticket.wait() == [[0, 0, 0, 2, 4, 5, 6, 103, 0, 0, 0, 204, 0, 0, 0, 2], updated])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

private let nestedHalfStorageShader = """
    enable f16;
    struct Cell { halves: array<f16, 2>, @size(4) marker: f16, }
    @group(0) @binding(0) var<storage, read_write> data: array<Cell>;

    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let row = lane / 2u;
        let item = lane % 2u;
        data[row].halves[item] += f16(lane + 1u);
    }
    """

private func halfPair(_ low: Double, _ high: Double) -> UInt32 {
    UInt32(Binary16.encode(low)) | (UInt32(Binary16.encode(high)) << 16)
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesNestedHalfStorageBuffer(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(nestedHalfStorageShader), to: format)
    #expect(shader.requiresShaderF16)
    #expect(shader.storageBindings.first?.runtimeArrayStride == 8)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(generated), capabilities: .referenceDefault.union(.shaderFloat16))
    }
    if format == .glsl { #expect(generated.contains("atomicCompSwap(")) }
    if format == .hlsl { #expect(generated.contains(".Store<half>(")) }
}

#if canImport(Metal)
@Test func nestedHalfStorageKeepsAdjacentWritesIndependentOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(nestedHalfStorageShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let data = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: data, offset: 0, words: [halfPair(1, 2), halfPair(9, -7), halfPair(3, 4), halfPair(10, -8)]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data], workgroups: .init(x: 1)),
            .copy(source: data, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[halfPair(2, 4), halfPair(9, -7), halfPair(6, 8), halfPair(10, -8)]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
