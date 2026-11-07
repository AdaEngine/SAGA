import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let sharedMemoryShader = """
    var<workgroup> scratch: array<u32, 7>;
    @group(0) @binding(2) var<storage, read_write> output: array<u32>;
    var<workgroup> flag: u32;
    fn fill(lane: u32, value: u32) { scratch[lane] = value; }
    fn sync() { workgroupBarrier(); }
    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
        let initial = scratch[lane + 3u] + flag;
        workgroupBarrier();
        fill(lane, group.x * 100u + lane + 10u);
        if lane == 0u { flag = 7u; }
        sync();
        output[group.x * 4u + lane] = initial * 1000u + scratch[(lane + 1u) % 4u] + flag;
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsWorkgroupMemoryAndBarriers(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(sharedMemoryShader), to: format)
    #expect(shader.storageBindings.map(\.binding.binding) == [2])
    #expect(shader.workgroupStorageByteSize == 48)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected text shader"); return }
    switch format {
    case .msl:
        #expect(source.contains("threadgroup uint saga_shared_0[7]"))
        #expect(source.contains("threadgroup uint* saga_shared_0"))
        #expect(source.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"))
    case .hlsl:
        #expect(source.contains("groupshared uint saga_shared_0[7]"))
        #expect(source.contains("GroupMemoryBarrierWithGroupSync()"))
    case .glsl:
        #expect(source.contains("shared uint saga_shared_0[7]"))
        #expect(source.contains("barrier()"))
        _ = try compiler.parseAndValidate(.glsl(source, stage: .compute, defines: [:]))
    case .wgsl:
        #expect(source.contains("var<workgroup> saga_shared_0: array<u32, 7>"))
        #expect(source.contains("workgroupBarrier()"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    "if lane == 0u { workgroupBarrier(); }",
    "if lane == 0u { sync(); }",
    "if lane == 0u { return; } workgroupBarrier();",
    "loop { if lane == 0u { return; } break; } workgroupBarrier();",
])
func rejectsBarriersSomeInvocationsCanSkip(body: String) {
    let source = """
        fn sync() { workgroupBarrier(); }
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) { \(body) }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Accepted a divergent barrier")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.contains { $0.code == .nonUniformControlFlow })
    } catch {
        Issue.record("Unexpected validation failure: \(error)")
    }
}

@Test func allowsBarriersAfterLoopReconvergenceAndUnderUniformGroupID() throws {
    let source = """
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            var i = 0u;
            loop { if i >= lane { break; } i += 1u; }
            workgroupBarrier();
            if group.x != 0u { workgroupBarrier(); }
        }
        """
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
}

@Test func rejectsExcessiveWorkgroupMemory() {
    let source = """
        var<workgroup> data: array<u32, 4096>;
        var<workgroup> tail: u32;
        @compute @workgroup_size(1) fn run() { data[0] = tail; }
        """
    #expect(throws: ShaderFailure.unsupported("Compute code generation: workgroup storage exceeds 16 KiB")) {
        try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl)
    }
}

@Test func workgroupInitializationSuppliesAnInvocationIndexWhenNeeded() throws {
    let source = """
        var<workgroup> value: u32;
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(1) fn run() { output = value; }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .msl)
    guard case .text(.msl, let output) = shader.artifact else { Issue.record("Expected MSL"); return }
    #expect(output.contains("uint saga_init_lane [[thread_index_in_threadgroup]]"))
    #expect(output.contains("saga_shared_0[0] = 0u"))
}

@Test func aUserFunctionCanShadowWorkgroupBarrier() throws {
    let source = """
        fn workgroupBarrier() {}
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32) {
            if lane == 0u { workgroupBarrier(); }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl)
    guard case .text(.msl, let output) = shader.artifact else { Issue.record("Expected MSL"); return }
    #expect(!output.contains("threadgroup_barrier"))
}

#if canImport(Metal)
@Test func workgroupMemoryStartsAtZeroAndSharesAcrossInvocationsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(sharedMemoryShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let buffer = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): buffer], workgroups: .init(x: 3)),
            .copy(source: buffer, destination: readback, byteCount: 48),
            .read(buffer: readback, byteCount: 48),
        ])
        #expect(try await ticket.wait() == [[18, 19, 20, 17, 118, 119, 120, 117, 218, 219, 220, 217]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
