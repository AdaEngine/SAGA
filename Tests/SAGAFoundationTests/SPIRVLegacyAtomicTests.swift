import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func legacyAtomicRuntimeArrayModule(atomicOnUniform: Bool = false, nonWritable: Bool = false) -> [UInt32] {
        var words = head
        words[1] = 0x00010000
        words += i(17, 1) + i(14, 0, 1) + i(15, 5, 30, 0x6e69616d, 0) + i(16, 30, 17, 1, 1, 1)
        if nonWritable { words += i(71, 12, 24) }
        let target: UInt32 = atomicOnUniform ? 40 : 43
        for instruction in [
            i(71, 3, 6, 4), i(71, 4, 3), i(72, 4, 0, 35, 0), i(71, 5, 2), i(72, 5, 0, 35, 0),
            i(71, 6, 3), i(72, 6, 0, 35, 0), i(72, 6, 1, 35, 4), i(72, 6, 2, 35, 8),
            i(71, 12, 34, 0), i(71, 12, 33, 0), i(71, 13, 34, 0), i(71, 13, 33, 1),
            i(71, 14, 34, 0), i(71, 14, 33, 2),
            i(19, 1), i(21, 2, 32, 0), i(29, 3, 2), i(30, 4, 3), i(30, 5, 2), i(30, 6, 2, 2, 2),
            i(32, 7, 2, 4), i(32, 8, 2, 5), i(32, 9, 2, 6), i(32, 10, 2, 2), i(33, 11, 1),
            i(43, 2, 20, 0), i(43, 2, 21, 1), i(43, 2, 22, 2),
            i(43, 2, 24, 0x00ff0000), i(43, 2, 25, 0x11ff3344), i(43, 2, 26, 0x55ff3344),
            i(59, 7, 12, 2), i(59, 8, 13, 2), i(59, 9, 14, 2), i(54, 1, 30, 0, 11), i(248, 31),
            i(65, 10, 40, 13, 20), i(61, 2, 41, 40), i(65, 10, 42, 12, 20, 41), i(83, 10, 43, 42),
            i(241, 2, 44, target, 21, 20, 24), i(230, 2, 45, target, 21, 20, 20, 26, 25),
            i(227, 2, 46, target, 21, 20),
            i(65, 10, 47, 14, 20), i(62, 47, 44), i(65, 10, 48, 14, 21), i(62, 48, 45),
            i(65, 10, 49, 14, 22), i(62, 49, 46), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvLegacyAtomicRejectsOrdinaryUniformPointer() throws {
        try fails(Self.legacyAtomicRuntimeArrayModule(atomicOnUniform: true), code: .unsupportedConstruct)
    }

    @Test func spirvLegacyAtomicRejectsReadModifyWriteOnNonWritableStorage() throws {
        try fails(Self.legacyAtomicRuntimeArrayModule(nonWritable: true))
    }
}

@Test func spirvLegacyAtomicRuntimeArrayCompilesToMSLAndWebGPUWGSL() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.legacyAtomicRuntimeArrayModule())
    let msl = try compiler.compileCompute(input, to: .msl)
    let wgsl = try compiler.compileWebGPUCompute(input)
    for artifact in [msl, wgsl] {
        #expect(artifact.storageBindings.map(\.binding) == [IRResourceBinding(group: 0, binding: 0), .init(group: 0, binding: 2)])
        #expect(artifact.uniformBindings.map(\.binding) == [IRResourceBinding(group: 0, binding: 1)])
        #expect(artifact.storageBindings.first?.runtimeArrayStride == 4)
    }
    guard case .text(.wgsl, let source) = wgsl.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("atomicOr("))
    #expect(source.contains("atomicCompareExchangeWeak("))
    _ = try compiler.parseAndValidate(.wgsl(source))
}

#if canImport(Metal)
@Test func spirvLegacyAtomicRuntimeArrayPreservesOriginalWordsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.legacyAtomicRuntimeArrayModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let packed = try await device.createBuffer(.init(size: 12, usage: [.storage, .copySource, .copyDestination]))
        let index = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 12, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: packed, offset: 0, words: [0xabcdef98, 0x11223344, 0x55667788]),
            .write(buffer: index, offset: 0, words: [1]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): packed, .init(binding: 1): index,
                                                             .init(binding: 2): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 12), .read(buffer: read, byteCount: 12),
            .copy(source: packed, destination: read, byteCount: 12), .read(buffer: read, byteCount: 12),
        ]).wait()
        #expect(snapshots == [[0x11223344, 0x11ff3344, 0x55ff3344], [0xabcdef98, 0x55ff3344, 0x55667788]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
