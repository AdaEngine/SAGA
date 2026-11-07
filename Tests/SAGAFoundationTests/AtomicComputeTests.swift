import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let atomicShader = """
    @group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    var<workgroup> local: atomic<i32>;
    fn ticket() -> u32 { return atomicAdd(&counter[0u], 1u); }
    @compute @workgroup_size(64)
    fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
        if lane == 0u { atomicStore(&local, -10i); }
        workgroupBarrier();
        let old = atomicAdd(&local, 1i);
        let global = ticket();
        let outside = atomicAdd(&counter[group.x + 1u], 1u);
        workgroupBarrier();
        let offset = 4u * (group.x * 64u + lane);
        output[offset] = global;
        output[offset + 1u] = u32(old);
        output[offset + 2u] = u32(atomicLoad(&local));
        output[offset + 3u] = outside;
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsAtomicComputeOperations(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(atomicShader), to: format)
    #expect(shader.storageBindings.map(\.runtimeArrayStride) == [4, 4])
    #expect(shader.workgroupStorageByteSize == 16)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .msl:
        #expect(source.contains("device atomic_uint*") && source.contains("threadgroup atomic_int"))
        #expect(source.contains("atomic_fetch_add_explicit") && source.contains("memory_order_relaxed"))
    case .hlsl:
        #expect(source.contains("InterlockedAdd(") && source.contains("InterlockedExchange("))
    case .glsl:
        #expect(source.contains("atomicAdd(") && source.contains("atomicExchange("))
    case .wgsl:
        #expect(source.contains("array<atomic<u32>>") && source.contains("atomicLoad("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: ["atomicAdd(&values, 1u)", "atomicAdd(values, 1u)", "atomicAdd(&counter, 1i)"])
func rejectsInvalidAtomicOperands(call: String) {
    let source = """
        var<workgroup> values: u32;
        var<workgroup> counter: atomic<u32>;
        @compute @workgroup_size(1) fn run() { _ = \(call); }
        """
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func atomicsSerializeContendedUpdatesAndProtectBindingRangesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(atomicShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let counter = try await device.createBuffer(.init(size: 264, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 2048, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 2048, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: counter, offset: 256, words: [0, 999]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0): .init(buffer: counter, offset: 256, size: 4),
                .init(binding: 1): .init(buffer: output),
            ], workgroups: .init(x: 2)),
            .copy(source: output, destination: readback, byteCount: 2048),
            .read(buffer: readback, byteCount: 2048),
            .copy(source: counter, destination: readback, byteCount: 264),
            .read(buffer: readback, byteCount: 264),
        ])
        let result = try await ticket.wait()
        try #require(result.count == 2 && result[0].count == 512)
        let words = result[0]
        let slots = stride(from: 0, to: words.count, by: 4)
        #expect(slots.map { words[$0] }.sorted() == Array(0..<128))
        #expect(slots.allSatisfy { words[$0 + 2] == 54 && words[$0 + 3] == 0 })
        for start in [0, 256] {
            let lanes = stride(from: start, to: start + 256, by: 4)
            #expect(lanes.map { Int32(bitPattern: words[$0 + 1]) }.sorted() == Array(-10..<54))
        }
        #expect(result[1] == Array(repeating: UInt32(0), count: 64) + [128, 999])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
