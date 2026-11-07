import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let mixedAtomicShader = """
    struct Mixed {
        @size(12) bias: u32,
        count: atomic<i32>,
        payload: vec2f,
        @align(32) values: array<atomic<u32>>,
    }
    @group(0) @binding(0) var<storage, read_write> data: Mixed;
    @group(0) @binding(1) var<storage, read_write> output: array<u32, 6>;
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        output[0] = u32(atomicMin(&data.count, -20i));
        output[1] = u32(atomicMax(&data.count, -6i));
        atomicOr(&data.count, 0i);
        output[2] = u32(atomicLoad(&data.count));
        data.bias += atomicAdd(&data.values[0u], 2u);
        let value = data.payload;
        data.payload = vec2f(value.y, value.x);
        output[3] = arrayLength(&data.values);
        let exchanged = atomicCompareExchangeWeak(&data.values[1u], 4u, 11u);
        if exchanged.exchanged { output[4] = 1u; }
        let outside = atomicCompareExchangeWeak(&data.values[id.x + arrayLength(&data.values)], 0u, 888u);
        output[5] = outside.old_value;
        if outside.exchanged { output[5] = 1u; }
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func preservesMixedAtomicStorageLayout(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(mixedAtomicShader), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64, 24])
    #expect(shader.storageBindings.map(\.runtimeArrayOffset) == [32, 0])
    #expect(shader.storageBindings.map(\.runtimeArrayStride) == [4, nil])
    #expect(shader.storageBindings.map(\.hlslUsesByteAddressBuffer) == [format == .hlsl, false])
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .msl:
        #expect(source.contains("device atomic_int*") && source.contains("device atomic_uint*"))
    case .hlsl:
        #expect(source.contains("RWByteAddressBuffer saga_buffer_0"))
        #expect(source.contains("saga_buffer_0.Load(") && source.contains("saga_buffer_0.Store("))
        #expect(source.contains("saga_buffer_0.InterlockedCompareExchange("))
        #expect(source.contains("asuint(min(asint(") && source.contains("asuint(max(asint("))
    case .glsl:
        #expect(source.contains("layout(offset = 12) int") && source.contains("layout(offset = 32) uint"))
    case .wgsl:
        let module = try compiler.parseAndValidate(.wgsl(source)).module
        let global = try #require(module.globalVariables.values.first)
        guard case .structure(let fields, let span) = try module.types.value(for: global.type).kind else {
            Issue.record("Expected storage structure"); return
        }
        #expect(fields.map(\.offset) == [0, 12, 16, 32])
        #expect(span == 64)
    default: Issue.record("Unexpected target")
    }
}

@Test func preservesMinimumBindingSizeOfFixedAtomicStructure() throws {
    let source = """
        struct Fixed { @align(16) count: atomic<u32>, tail: u32 }
        @group(0) @binding(0) var<storage, read_write> data: Fixed;
        @compute @workgroup_size(1) fn run() { atomicStore(&data.count, data.tail); }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    #expect(shader.storageBindings.first?.minimumByteSize == 16)
    guard case .text(.wgsl, let output) = shader.artifact else { Issue.record("Expected WGSL"); return }
    let module = try compiler.parseAndValidate(.wgsl(output)).module
    let global = try #require(module.globalVariables.values.first)
    guard case .structure(let fields, let span) = try module.types.value(for: global.type).kind else {
        Issue.record("Expected storage structure"); return
    }
    #expect(fields.map(\.offset) == [0, 4])
    #expect(span == 16)
}

#if canImport(Metal)
@Test func runsAtomicStorageFieldsThroughRangedBindingsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(mixedAtomicShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let data = try await device.createBuffer(.init(size: 324, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 24, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 324, usage: [.copyDestination, .mapRead]))
        let initial: [UInt32] = [9, 701, 702, UInt32(bitPattern: -10), Float(1.5).bitPattern, Float(-2).bitPattern,
                                703, 704, 3, 4, 0, 0, 0, 0, 0, 0, 999]
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([
                .dispatchComputeRanges(pipeline: pipeline, bindings: [
                    .init(binding: 0): .init(buffer: data, offset: 256, size: 60),
                    .init(binding: 1): .init(buffer: output),
                ], workgroups: .init(x: 1)),
            ])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: data, offset: 256, words: initial),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0): .init(buffer: data, offset: 256, size: 64),
                .init(binding: 1): .init(buffer: output),
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 24),
            .read(buffer: read, byteCount: 24),
            .copy(source: data, destination: read, byteCount: 324),
            .read(buffer: read, byteCount: 324),
        ])
        let result = try await ticket.wait()
        try #require(result.count == 2 && result[0].count == 6)
        let exchanged = result[0][4]
        #expect(exchanged <= 1)
        #expect(result[0] == [UInt32(bitPattern: -10), UInt32(bitPattern: -20), UInt32(bitPattern: -6), 8, exchanged, 0])
        var expected = initial
        expected[0] = 12
        expected[3] = UInt32(bitPattern: -6)
        expected.swapAt(4, 5)
        expected[8] = 5
        expected[9] = exchanged == 1 ? 11 : 4
        #expect(result[1] == Array(repeating: UInt32(0), count: 64) + expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
