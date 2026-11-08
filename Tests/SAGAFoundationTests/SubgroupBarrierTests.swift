import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let subgroupBarrierOnly = "@compute @workgroup_size(1) fn run() { subgroupBarrier(); }"
private let subgroupBarrierExchange = """
    var<workgroup> scratch: array<u32, 64>;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 64>;
    fn synchronize() { subgroupBarrier(); }
    @compute @workgroup_size(64) fn run(@builtin(local_invocation_index) lane: u32) {
        scratch[lane] = lane + 100u;
        synchronize();
        output[lane] = scratch[lane ^ 1u];
    }
    """

extension SPIRVFrontendTests {
    static func subgroupBarrierModule(memoryScope: UInt32 = 3, semantics: UInt32 = 0x88) -> [UInt32] {
        pre + i(15, 5, 6, 0x6e69616d, 0) + i(16, 6, 17, 1, 1, 1)
            + i(19, 1) + i(21, 2, 32, 0) + i(33, 3, 1)
            + i(43, 2, 4, 3) + i(43, 2, 5, memoryScope) + i(43, 2, 8, semantics)
            + i(54, 1, 6, 0, 3) + i(248, 7) + i(224, 4, 5, 8) + i(253) + i(56)
    }

    @Test func subgroupBarrierSPIRVChecksScopeAndPreservesMemoryFlags() throws {
        try fails(Self.subgroupBarrierModule(memoryScope: 2), code: .unsupportedConstruct)
        let module = try parse(Self.subgroupBarrierModule(semantics: 0xd48))
        let barriers = module.functions.values.flatMap { function in
            function.statements.values.compactMap { statement -> IRBarrier? in
                if case .controlBarrier(let flags) = statement.kind { return flags }
                return nil
            }
        }
        #expect(barriers == [[.subgroup, .workgroup, .storage, .texture]])
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .glsl])
func subgroupBarrierRetainsExecutionScopeAcrossInputs(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for input in [ShaderInput.wgsl(subgroupBarrierOnly), .spirv(SPIRVFrontendTests.subgroupBarrierModule())] {
        let shader = try compiler.compileCompute(input, to: format)
        #expect(shader.requiresSubgroups)
        guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.invalidHandle }
        switch format {
        case .msl:
            #expect(text.contains("simdgroup_barrier(mem_flags::mem_threadgroup)"))
            #expect(!text.contains("threadgroup_barrier("))
        case .glsl:
            #expect(text.contains("#extension GL_KHR_shader_subgroup_basic : require"))
            #expect(text.contains("subgroupBarrier();"))
        case .wgsl:
            #expect(text.contains("subgroupBarrier();"))
            _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.subgroup, .subgroupBarrier])
        default: Issue.record("Unexpected target")
        }
    }
}

@Test func subgroupBarrierRejectsProfilesWithoutASubgroupPrimitive() {
    let compiler = IndependentShaderCompiler()
    for source in [ShaderInput.wgsl(subgroupBarrierOnly), .spirv(SPIRVFrontendTests.subgroupBarrierModule())] {
        #expect(throws: ShaderFailure.self) { try compiler.compileCompute(source, to: .hlsl) }
        #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(source) }
    }
}

@Test func subgroupBarrierValidatesCapabilityAndUniformControlFlow() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(subgroupBarrierOnly)) }
    let divergent = ShaderInput.wgsl("""
        @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
            if lane == 0u { subgroupBarrier(); }
        }
        """)
    do {
        _ = try compiler.compileCompute(divergent, to: .msl)
        Issue.record("Accepted a subgroup barrier in divergent control flow")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .nonUniformControlFlow)
    }
}

#if canImport(Metal)
@Test func subgroupBarrierSynchronizesPeerWritesOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(subgroupBarrierExchange), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 256, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 256, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 256), .read(buffer: readback, byteCount: 256),
        ]).wait()
        let expected: [UInt32] = (0..<64).map { (lane: Int) in UInt32((lane ^ 1) + 100) }
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
