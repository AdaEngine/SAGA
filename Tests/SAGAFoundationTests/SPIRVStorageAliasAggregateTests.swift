import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func aggregateAliasSPIRV() -> ShaderInput {
    let i = SPIRVFrontendTests.i
    var words = SPIRVFrontendTests.pre
    // A pair of sign-reinterpreted storage blocks at binding zero, plus a copy destination.
    words += i(15, 5, 30, 0x6e69616d, 0) + i(16, 30, 17, 1, 1, 1)
    words += i(71, 4, 6, 4) + i(71, 5, 6, 4) + i(71, 6, 2) + i(71, 7, 2)
    words += i(72, 6, 0, 35, 0) + i(72, 7, 0, 35, 0)
    for (variable, binding): (UInt32, UInt32) in [(20, 0), (21, 0), (22, 1)] {
        words += i(71, variable, 34, 0) + i(71, variable, 33, binding)
        if binding == 0 { words += i(71, variable, 20) }
    }
    for instruction in [
        i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(43, 2, 11, 2),
        i(28, 4, 2, 11), i(28, 5, 3, 11), i(30, 6, 4), i(30, 7, 5), i(32, 8, 12, 6), i(32, 9, 12, 7), i(33, 10, 1),
        i(43, 3, 12, 0xfffffffd), i(43, 3, 13, 0xfffffff7), i(44, 5, 14, 12, 13), i(44, 7, 15, 14),
        i(59, 8, 20, 12), i(59, 9, 21, 12), i(59, 8, 22, 12), i(54, 1, 30, 0, 10), i(248, 31),
        i(62, 21, 15), i(61, 6, 32, 20), i(62, 22, 32), i(253), i(56),
    ] { words += instruction }
    return .spirv(words)
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvStorageAliasRebuildsWholeStructAndArray(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(aggregateAliasSPIRV(), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [8, 8])
    if case .text(.wgsl, let code) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(code))
    }
}

#if canImport(Metal)
@Test func spirvStorageAliasWritesEntireAggregateOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(aggregateAliasSPIRV(), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let shared = try await device.createBuffer(.init(size: 8, usage: [.storage]))
        let output = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): shared, .init(binding: 1): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[0xfffffffd, 0xfffffff7]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
