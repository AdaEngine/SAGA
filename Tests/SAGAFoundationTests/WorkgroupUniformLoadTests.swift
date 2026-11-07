import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let uniformLoadShader = """
    var<workgroup> flag: u32;
    var<workgroup> values: array<vec2u, 2>;
    var<workgroup> counter: atomic<i32>;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 12>;
    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
        if lane == 0u {
            flag = 11u + group.x;
            values[group.x] = vec2u(7u + group.x, 2u);
            atomicStore(&counter, -5i + i32(group.x));
        }
        let snapshot = workgroupUniformLoad(&flag);
        if lane == 0u { flag = 666u; }
        let next = workgroupUniformLoad(&flag);
        if snapshot != 0u { workgroupBarrier(); }
        let pair = workgroupUniformLoad(&values[group.x]);
        if pair.x != 0u { workgroupBarrier(); }
        let signedValue = workgroupUniformLoad(&counter);
        if signedValue < 0i { workgroupBarrier(); }
        output[group.x * 4u + lane] = snapshot + pair.x + pair.y + u32(signedValue + 10i) + next;
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func lowersUniformWorkgroupLoadsAndAtomicOverload(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let compiled = try compiler.compileCompute(.wgsl(uniformLoadShader), to: format)
    #expect(compiled.workgroupStorageByteSize == 48)
    guard case .text(_, let source) = compiled.artifact else { Issue.record("Expected shader source"); return }
    func barriers(_ spelling: String) -> Int { source.split(separator: "\n").filter { $0.contains(spelling) }.count }
    switch format {
    case .msl:
        #expect(barriers("threadgroup_barrier(mem_flags::mem_threadgroup)") == 12)
        #expect(source.contains("atomic_load_explicit"))
    case .hlsl:
        #expect(barriers("GroupMemoryBarrierWithGroupSync()") == 12)
        #expect(source.contains("InterlockedAdd("))
    case .glsl:
        #expect(barriers("memoryBarrierShared();") == 12 && barriers("barrier();") == 12)
        #expect(source.contains("atomicAdd("))
    case .wgsl:
        #expect(source.contains("workgroupUniformLoad(&"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    "if lane == 0u { _ = workgroupUniformLoad(&values[0u]); }",
    "_ = workgroupUniformLoad(&values[lane]);",
    "if lane == 0u { _ = readUniform(); }",
])
func rejectsDivergentWorkgroupUniformLoads(body: String) {
    let source = """
        var<workgroup> values: array<u32, 4>;
        fn readUniform() -> u32 { return workgroupUniformLoad(&values[0u]); }
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) { \(body) }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Accepted divergent workgroup uniform load")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.contains { $0.code == .nonUniformControlFlow })
    } catch {
        Issue.record("Unexpected validation error: \(error)")
    }
}

@Test func rejectsNonWorkgroupUniformLoadPointer() {
    let source = """
        @group(0) @binding(0) var<storage, read_write> value: u32;
        @compute @workgroup_size(1) fn run() { _ = workgroupUniformLoad(&value); }
        """
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func workgroupUniformLoadsSynchronizeBeforeAndAfterReadOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(uniformLoadShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 3)),
            .copy(source: output, destination: read, byteCount: 48),
            .read(buffer: read, byteCount: 48),
        ])
        #expect(try await ticket.wait() == [[691, 691, 691, 691, 694, 694, 694, 694, 686, 686, 686, 686]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
