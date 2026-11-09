import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvUniformLoadInput = ShaderInput.wgsl("""
    var<workgroup> values: array<u32, 2>;
    var<workgroup> counters: array<atomic<i32>, 2>;
    @group(0) @binding(0) var<storage, read_write> output: array<vec2u, 4>;

    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32,
                                     @builtin(workgroup_id) group: vec3u) {
        if lane == 0u {
            values[group.x] = group.x + 1u;
            atomicStore(&counters[group.x], -2i + i32(group.x));
        }
        let value = workgroupUniformLoad(&values[group.x]);
        if value != 0u { workgroupBarrier(); }
        let counter = workgroupUniformLoad(&counters[group.x]);
        if counter < 0i { workgroupBarrier(); }
        output[group.x * 2u + lane] = vec2u(value, bitcast<u32>(counter));
    }
    """)

private func exportedUniformLoad(_ compiler: IndependentShaderCompiler) throws -> ShaderInput {
    guard case .spirv(let words) = try compiler.compile(spirvUniformLoadInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return .spirv(words)
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvWorkgroupUniformityRecoversOrdinaryAndAtomicCollectiveLoads(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(exportedUniformLoad(compiler), to: format)
    #expect(shader.workgroupStorageByteSize == 32)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [32])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

extension SPIRVFrontendTests {
    // The two invocations write and reload distinct elements. Barriers do not make the values uniform.
    static func varyingWorkgroupLoadModule(dependentBarrier: Bool) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 40, 0x6e69616d, 0, 30), i(16, 40, 17, 2, 1, 1),
            i(71, 30, 11, 29), i(71, 5, 6, 4), i(71, 6, 2), i(72, 6, 0, 35, 0),
            i(71, 32, 34, 0), i(71, 32, 33, 0),
            i(19, 1), i(20, 2), i(21, 3, 32, 0), i(43, 3, 20, 0), i(43, 3, 21, 2), i(43, 3, 22, 0x108),
            i(28, 4, 3, 21), i(28, 5, 3, 21), i(30, 6, 5), i(32, 7, 1, 3), i(32, 8, 4, 4),
            i(32, 9, 4, 3), i(32, 10, 12, 6), i(32, 11, 12, 3), i(33, 12, 1),
            i(59, 7, 30, 1), i(59, 8, 31, 4), i(59, 10, 32, 12),
            i(54, 1, 40, 0, 12), i(248, 41), i(61, 3, 42, 30), i(65, 9, 43, 31, 42), i(62, 43, 42),
            i(224, 21, 21, 22), i(61, 3, 44, 43), i(224, 21, 21, 22),
        ] { words += instruction }
        if dependentBarrier {
            words += i(170, 2, 45, 44, 20) + i(247, 47, 0) + i(250, 45, 46, 47)
            words += i(248, 46) + i(224, 21, 21, 22) + i(249, 47) + i(248, 47)
        }
        return words + i(65, 11, 48, 32, 20, 42) + i(62, 48, 44) + i(253) + i(56)
    }

    static var mixedWorkgroupLoadCallersModule: [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 40, 0x6e69616d, 0, 30), i(16, 40, 17, 2, 1, 1),
            i(71, 30, 11, 29), i(71, 5, 6, 4), i(71, 6, 2), i(72, 6, 0, 35, 0), i(71, 32, 34, 0), i(71, 32, 33, 0),
            i(19, 1), i(21, 3, 32, 0), i(43, 3, 20, 0), i(43, 3, 21, 2), i(43, 3, 22, 0x108), i(43, 3, 23, 1),
            i(28, 4, 3, 21), i(28, 5, 3, 21), i(30, 6, 5), i(32, 7, 1, 3), i(32, 8, 4, 4),
            i(32, 9, 4, 3), i(32, 10, 12, 6), i(32, 11, 12, 3), i(33, 12, 1), i(33, 13, 3, 3),
            i(59, 7, 30, 1), i(59, 8, 31, 4), i(59, 10, 32, 12),
            // Helper: return the workgroup element at the supplied index, with barriers around the load.
            i(54, 3, 50, 0, 13), i(55, 3, 51), i(248, 52), i(65, 9, 53, 31, 51),
            i(224, 21, 21, 22), i(61, 3, 54, 53), i(224, 21, 21, 22), i(254, 54), i(56),
            i(54, 1, 40, 0, 12), i(248, 41), i(61, 3, 42, 30), i(65, 9, 43, 31, 42), i(62, 43, 42),
            i(57, 3, 44, 50, 23), i(57, 3, 45, 50, 42), i(128, 3, 46, 44, 45),
            i(65, 11, 47, 32, 20, 42), i(62, 47, 46), i(253), i(56),
        ] { words += instruction }
        return words
    }
}

@Test func spirvWorkgroupUniformityRejectsDependentBarrierAfterVaryingLoad() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.varyingWorkgroupLoadModule(dependentBarrier: true))
    do {
        _ = try compiler.parseAndValidate(input)
        Issue.record("Accepted a barrier dependent on a varying workgroup load")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .nonUniformControlFlow })
    }
}

@Test func spirvWorkgroupUniformityKeepsLegalVaryingLoadOrdinary() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.varyingWorkgroupLoadModule(dependentBarrier: false))
    let shader = try compiler.compileCompute(input, to: .wgsl)
    guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(!text.contains("workgroupUniformLoad("))
    _ = try compiler.parseAndValidate(.wgsl(text))
}

#if canImport(Metal)
@Test func spirvWorkgroupUniformityKeepsMixedCallSiteLoadsOrdinaryOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.mixedWorkgroupLoadCallersModule)
    let wgsl = try compiler.compileCompute(input, to: .wgsl)
    guard case .text(.wgsl, let text) = wgsl.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(!text.contains("workgroupUniformLoad("))
    _ = try compiler.parseAndValidate(.wgsl(text))
    let shader = try compiler.compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[1, 2]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func spirvWorkgroupUniformityPreservesCollectiveResultsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(exportedUniformLoad(compiler), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 2)),
            .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
        ]).wait()
        #expect(snapshots == [[1, 0xfffffffe, 1, 0xfffffffe, 2, .max, 2, .max]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
