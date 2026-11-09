import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func storageBlockModule(
        legacy: Bool = true, storeToUniform: Bool = false, dualDecoration: Bool = false,
        legacyWithWrongClass: Bool = false, includeModernStorage: Bool = false
    ) -> [UInt32] {
        var words = head
        if legacy && !legacyWithWrongClass && !includeModernStorage { words[1] = 0x00010000 }
        let bufferClass: UInt32 = legacy && !legacyWithWrongClass ? 2 : 12
        let scalarPointer: UInt32 = bufferClass == 2 ? 8 : 14
        words += i(17, 1) + i(14, 0, 1) + i(15, 5, 20, 0x6e69616d, 0) + i(16, 20, 17, 1, 1, 1)
        for instruction in [
            i(71, 4, legacy ? 3 : 2), i(71, 5, 2),
            i(72, 3, 0, 35, 0), i(72, 3, 1, 35, 4), i(72, 4, 0, 35, 0), i(72, 5, 0, 35, 0),
            i(71, 10, 34, 0), i(71, 10, 33, 0), i(71, 11, 34, 0), i(71, 11, 33, 1),
        ] { words += instruction }
        if dualDecoration { words += i(71, 4, 2) }
        if includeModernStorage { words += i(71, 18, 34, 0) + i(71, 18, 33, 2) }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(30, 3, 2, 2), i(30, 4, 3), i(30, 5, 2),
            i(32, 6, bufferClass, 4), i(32, 7, 2, 5), i(32, 8, 2, 2), i(32, 15, bufferClass, 3), i(33, 9, 1),
            i(43, 2, 12, 0), i(43, 2, 13, 1),
        ] { words += instruction }
        if bufferClass == 12 { words += i(32, 14, 12, 2) }
        if includeModernStorage { words += i(32, 17, 12, 5) + i(59, 17, 18, 12) }
        for instruction in [
            i(59, 6, 10, bufferClass), i(59, 7, 11, 2), i(54, 1, 20, 0, 9), i(248, 21),
            i(65, 15, 30, 10, 12), i(65, scalarPointer, 31, 30, 13), i(83, scalarPointer, 32, 31), i(61, 2, 33, 32),
            i(65, 8, 34, 11, 12), i(61, 2, 35, 34), i(128, 2, 36, 33, 35),
            i(62, storeToUniform ? 34 : 32, 36), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvLegacyStorageRejectsInvalidDecorationClassAndUniformWrites() throws {
        try fails(Self.storageBlockModule(storeToUniform: true))
        try fails(Self.storageBlockModule(dualDecoration: true))
        try fails(Self.storageBlockModule(legacyWithWrongClass: true))
        try fails(Self.storageBlockModule(includeModernStorage: true))
    }

    @Test func spirvBufferBlockStopsAtVersion13() throws {
        var words = Self.storageBlockModule()
        words[1] = 0x00010300
        #expect(try parse(words).globalVariables.count == 2)

        words[1] = 0x00010400
        do {
            _ = try parse(words)
            Issue.record("Accepted BufferBlock in SPIR-V 1.4")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.first?.code == .invalidBinary)
            #expect(diagnostics.first?.message == "BufferBlock requires SPIR-V 1.3 or older")
        }
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvLegacyStorageReflectsSameBindingsAsModernStorage(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let legacy = try compiler.compileCompute(.spirv(SPIRVFrontendTests.storageBlockModule()), to: format)
    let modern = try compiler.compileCompute(.spirv(SPIRVFrontendTests.storageBlockModule(legacy: false)), to: format)
    #expect(legacy.storageBindings == modern.storageBindings)
    #expect(legacy.uniformBindings == modern.uniformBindings)
    #expect(legacy.storageBindings.map(\.binding) == [IRResourceBinding(group: 0, binding: 0)])
    #expect(legacy.uniformBindings.map(\.binding) == [IRResourceBinding(group: 0, binding: 1)])
    if format == .wgsl {
        _ = try compiler.compileWebGPUCompute(.spirv(SPIRVFrontendTests.storageBlockModule()))
    }
}

#if canImport(Metal)
@Test(arguments: [true, false])
func spirvLegacyStorageMatchesModernGPUReadback(_ legacy: Bool) async throws {
    let words = SPIRVFrontendTests.storageBlockModule(legacy: legacy)
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let storage = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource, .copyDestination]))
        let uniform = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: storage, offset: 0, words: [11, 22]),
            .write(buffer: uniform, offset: 0, words: [20]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): storage, .init(binding: 1): uniform],
                             workgroups: .init(x: 1)),
            .copy(source: storage, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[11, 42]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
