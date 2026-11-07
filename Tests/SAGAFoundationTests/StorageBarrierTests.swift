import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let storageBarrierShader = """
    @group(0) @binding(1) var<storage, read_write> scratch: array<u32>;
    @group(2) @binding(1) var<storage, read_write> output: array<u32>;
    fn syncDevice() { storageBarrier(); }
    fn rendezvous() { syncDevice(); }
    @compute @workgroup_size(128)
    fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
        let start = group.x * 128u;
        scratch[start + lane] = group.x * 1000u + lane + 10u;
        rendezvous();
        output[start + lane] = scratch[start + (lane + 64u) % 128u] + 7u;
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsStorageBarrierWithDeviceMemorySemantics(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(storageBarrierShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected text shader"); return }
    switch format {
    case .msl: #expect(source.contains("threadgroup_barrier(mem_flags::mem_device);"))
    case .hlsl: #expect(source.contains("DeviceMemoryBarrierWithGroupSync();"))
    case .glsl:
        #expect(source.contains("memoryBarrierBuffer();\n    barrier();"))
        let module = try compiler.parseAndValidate(.glsl(source, stage: .compute, defines: [:])).module
        let bindings = module.globalVariables.values.filter { $0.binding != nil }
        #expect(bindings.count == 2)
        #expect(bindings.allSatisfy { $0.memoryDecorations == .coherent })
    case .wgsl:
        #expect(source.contains("storageBarrier();"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target")
    }
}

@Test func unreachableStorageBarrierDoesNotChangeGLSLStorageQualifiers() throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> output: u32;
        fn unused() { storageBarrier(); }
        @compute @workgroup_size(1) fn run() { output = 1u; }
        """), to: .glsl)
    guard case .text(.glsl, let source) = shader.artifact else { Issue.record("Expected GLSL"); return }
    #expect(!source.contains("coherent"))
    #expect(!source.contains("memoryBarrierBuffer"))
}

@Test func rejectsStorageBarrierInDivergentTransitiveCall() {
    let source = """
        fn syncDevice() { storageBarrier(); }
        fn rendezvous() { syncDevice(); }
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32) {
            if lane != 0u { rendezvous(); }
        }
        """
    do {
        _ = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl)
        Issue.record("Accepted a divergent storage barrier")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.contains { $0.code == .nonUniformControlFlow })
    } catch {
        Issue.record("Unexpected validation failure: \(error)")
    }
}

#if canImport(Metal)
@Test func storageBarrierMakesNeighborWritesVisibleOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(storageBarrierShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let outputBytes: UInt64 = 3 * 128 * 4
        let sentinelOffset = 256 + outputBytes
        let scratchBytes = sentinelOffset + 4
        let scratch = try await device.createBuffer(.init(size: scratchBytes, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: outputBytes, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: scratchBytes, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: scratch, offset: sentinelOffset, words: [999]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 1): .init(buffer: scratch, offset: 256, size: outputBytes),
                .init(group: 2, binding: 1): .init(buffer: output),
            ], workgroups: .init(x: 3)),
            .copy(source: output, destination: readback, byteCount: outputBytes),
            .read(buffer: readback, byteCount: outputBytes),
            .copy(source: scratch, destination: readback, byteCount: scratchBytes),
            .read(buffer: readback, byteCount: scratchBytes),
        ])
        let writes = (0..<384).map { (index: Int) -> UInt32 in
            let group = UInt32(index / 128)
            let lane = UInt32(index % 128)
            return group * 1000 + lane + 10
        }
        let rotated = (0..<384).map { (index: Int) -> UInt32 in
            let start = (index / 128) * 128
            let lane = (index + 64) % 128
            return writes[start + lane] + 7
        }
        let scratchExpected = Array(repeating: UInt32(0), count: 64) + writes + [999]
        #expect(try await ticket.wait() == [rotated, scratchExpected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
