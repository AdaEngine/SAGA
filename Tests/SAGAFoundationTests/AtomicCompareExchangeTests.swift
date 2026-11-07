import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let atomicCompareShader = """
    @group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32, 10>;
    var<workgroup> signedValue: atomic<i32>;
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        atomicStore(&signedValue, -9i);
        var signedResult = atomicCompareExchangeWeak(&signedValue, -9i, 11i);
        output[0] = u32(signedResult.old_value);
        if signedResult.exchanged { output[1] = 1u; }
        output[2] = u32(atomicLoad(&signedValue));
        signedResult = atomicCompareExchangeWeak(&signedValue, 999i, 12i);
        output[3] = u32(signedResult.old_value);
        if signedResult.exchanged { output[4] = 1u; }
        let unsignedResult = atomicCompareExchangeWeak(&counter[0u], 0u, 7u);
        output[5] = unsignedResult.old_value;
        if unsignedResult.exchanged { output[6] = 1u; }
        output[7] = atomicLoad(&counter[0u]);
        let outside = atomicCompareExchangeWeak(&counter[id.x + 1u], 0u, 888u);
        output[8] = outside.old_value;
        if outside.exchanged { output[9] = 1u; }
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func lowersAtomicCompareExchangeWithTypedResults(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let validated = try compiler.parseAndValidate(.wgsl(atomicCompareShader))
    #expect(validated.module.specialTypes.predeclared.map(\.kind) == [
        .atomicCompareExchangeWeakResult(.i32), .atomicCompareExchangeWeakResult(.u32),
    ])
    let compiled = try compiler.compileCompute(.wgsl(atomicCompareShader), to: format)
    guard case .text(_, let source) = compiled.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .msl: #expect(source.contains("atomic_compare_exchange_weak_explicit("))
    case .hlsl: #expect(source.contains("InterlockedCompareExchange("))
    case .glsl: #expect(source.contains("atomicCompSwap("))
    case .wgsl:
        #expect(source.contains(".old_value") && source.contains(".exchanged"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    "atomicCompareExchangeWeak(&counter, 0i, 1u)",
    "atomicCompareExchangeWeak(&counter, 0u, 1i)",
    "atomicCompareExchangeWeak(&counter, 0u)",
])
func rejectsInvalidAtomicCompareExchangeOperands(call: String) {
    let source = """
        var<workgroup> counter: atomic<u32>;
        @compute @workgroup_size(1) fn run() { _ = \(call); }
        """
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func atomicCompareExchangeReportsOldValueAndSuccessOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(atomicCompareShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let counter = try await device.createBuffer(.init(size: 8, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 40, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 40, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: counter, offset: 4, words: [999]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0): .init(buffer: counter, size: 4),
                .init(binding: 1): .init(buffer: output),
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 40),
            .read(buffer: readback, byteCount: 40),
            .copy(source: counter, destination: readback, byteCount: 8),
            .read(buffer: readback, byteCount: 8),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2 && snapshots[0].count == 10)
        let result = snapshots[0]
        let signedStart = UInt32(bitPattern: -9)
        let signedAfter: UInt32 = result[1] == 1 ? 11 : signedStart
        let unsignedAfter: UInt32 = result[6] == 1 ? 7 : 0
        #expect(result[1] <= 1 && result[6] <= 1)
        #expect(result == [signedStart, result[1], signedAfter, signedAfter, 0, 0, result[6], unsignedAfter, 0, 0])
        #expect(snapshots[1] == [unsignedAfter, 999])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
