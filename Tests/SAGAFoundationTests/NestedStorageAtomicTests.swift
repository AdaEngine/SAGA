import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let nestedAtomicShader = """
    requires unrestricted_pointer_parameters;
    struct Cell { @size(16) signedValues: array<atomic<i32>, 2>, count: atomic<u32>, @size(12) tag: u32, }
    struct Data { @size(8) header: u32, cells: array<Cell>, }
    @group(0) @binding(0) var<storage, read_write> data: Data;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 8>;

    fn update(p: ptr<storage, atomic<i32>, read_write>) -> vec4i {
        let beforeMin = atomicMin(p, -20i);
        let beforeMax = atomicMax(p, -8i);
        let mismatch = atomicCompareExchangeWeak(p, -99i, 100i);
        var succeeded = false;
        loop {
            let attempt = atomicCompareExchangeWeak(p, -8i, -2i);
            succeeded = attempt.exchanged;
            if succeeded || attempt.old_value != -8i { break; }
        }
        let beforeSub = atomicSub(p, -4i);
        return vec4i(beforeMin, beforeMax, mismatch.old_value + select(0i, 100i, mismatch.exchanged),
                     beforeSub + select(0i, 100i, succeeded));
    }

    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let row = i32(lane) - 1i;
        let index = select(1u, 2u, lane == 2u);
        let values = update(&data.cells[row].signedValues[index]);
        let old = atomicAdd(&data.cells[row].count, lane + 1u);
        let beforeExchange = atomicExchange(&data.cells[row].count, old + 20u);
        output[lane * 2u] = bitcast<vec4u>(values);
        output[lane * 2u + 1u] = vec4u(old, beforeExchange, data.cells[row].tag, arrayLength(&data.cells));
        storageBarrier();
        atomicAdd(&data.cells[0].count, 1u);
    }
    """

private let nestedAtomicInput: [UInt32] = [
    7, 800,
    UInt32(bitPattern: -4), 5, 801, 802, 10, 100, 803, 804,
    UInt32(bitPattern: -5), UInt32(bitPattern: -6), 805, 806, 20, 200, 807, 808,
]

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesNestedStorageAtomics(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(nestedAtomicShader), to: format)
    #expect(shader.storageBindings.first?.minimumByteSize == 40)
    #expect(shader.storageBindings.first?.runtimeArrayOffset == 8)
    #expect(shader.storageBindings.first?.runtimeArrayStride == 32)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
    if format == .glsl {
        #expect(generated.contains("atomicCompSwap("))
        #expect(generated.contains("uint(min(int(") && generated.contains("uint(max(int("))
    }
}

@Test func rejectsCopyOfNestedAtomicValue() {
    let source = ShaderInput.wgsl("""
        struct Cell { values: array<atomic<u32>, 2>, }
        @group(0) @binding(0) var<storage, read_write> data: array<Cell>;
        @compute @workgroup_size(1) fn run() { let cell = data[0]; }
        """)
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(source) }
}

#if canImport(Metal)
@Test func nestedStorageAtomicsPreserveSignsAndPaddingOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(nestedAtomicShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let data = try await device.createBuffer(.init(size: 72, usage: [.storage, .copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: data, offset: 0, words: nestedAtomicInput),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
            .copy(source: data, destination: read, byteCount: 72), .read(buffer: read, byteCount: 72),
        ])
        var expected = nestedAtomicInput
        expected[3] = 2
        expected[6] = 34
        expected[14] = 40
        #expect(try await ticket.wait() == [[
            0, 0, 0, 0, 0, 0, 0, 2,
            5, UInt32(bitPattern: -20), UInt32(bitPattern: -8), 98, 10, 12, 100, 2,
            0, 0, 0, 0, 20, 23, 200, 2,
            0, 0, 0, 0, 0, 0, 0, 2,
        ], expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
