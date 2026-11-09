import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

extension SPIRVFrontendTests {
    private static func scalarSpecializationCompute(id: UInt32 = 100, numericOpcode: UInt32 = 50) -> [UInt32] {
        var words = head + i(17, 1) + i(14, 0, 1)
        words += i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        for instruction in [
            i(71, 10, 1, id), i(71, 11, 1, 101), i(71, 12, 1, 102), i(71, 5, 2),
            i(72, 5, 0, 35, 0), i(72, 5, 1, 35, 4), i(71, 20, 33, 0), i(71, 20, 34, 0),
            i(19, 1), i(21, 2, 32, 0), i(20, 3), i(30, 5, 2, 2),
            i(32, 6, 12, 5), i(32, 7, 12, 2), i(33, 8, 1),
            i(numericOpcode, 2, 10, 7), i(48, 3, 11), i(49, 3, 12),
            i(43, 2, 13, 0), i(43, 2, 14, 1), i(43, 2, 15, 3), i(43, 2, 16, 19),
            i(59, 6, 20, 12), i(54, 1, 40, 0, 8), i(248, 41),
            i(169, 2, 42, 11, 10, 15), i(169, 2, 43, 12, 10, 16),
            i(65, 7, 44, 20, 13), i(65, 7, 45, 20, 14), i(62, 44, 42), i(62, 45, 43),
            i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
    func spirvScalarSpecializationKeepsIDsAndSpecializesBeforeTextEmission(format: ShaderOutputFormat) throws {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.spirv(Self.scalarSpecializationCompute())
        let module = try compiler.parseAndValidate(source).module
        #expect(module.overrides.values.map(\.id) == [100, 101, 102])
        let defaults = try compiler.compileCompute(source, to: format)
        let specialized = try compiler.compileCompute(source, to: format, constants: ["100": 25, "101": 0, "102": 1])
        #expect(defaults.artifact != specialized.artifact)
        #expect(specialized.storageBindings.map(\.minimumByteSize) == [8])
    }

    @Test func spirvScalarSpecializationRejectsInvalidSpecIdTargetsAndRange() throws {
        try fails(Self.scalarSpecializationCompute(numericOpcode: 43))
        try fails(Self.scalarSpecializationCompute(id: 65_536), code: .unsupportedConstruct)
    }

    #if canImport(Metal)
    @Test func spirvScalarSpecializationUsesDefaultsAndSuppliedValuesOnMetal() async throws {
        let source = ShaderInput.spirv(Self.scalarSpecializationCompute())
        let compiler = IndependentShaderCompiler()
        let device = try await Device.make(backend: MetalBackend.makeDevice())
        do {
            let output = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
            let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
            for (constants, expected): ([String: Double], [UInt32]) in [([:], [7, 19]), (["100": 25, "101": 0, "102": 1], [3, 25])] {
                let shader = try compiler.compileCompute(source, to: .msl, constants: constants)
                let pipeline = try await device.createComputePipeline(shader)
                let snapshots = try await device.submitCommands([
                    .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
                    .copy(source: output, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
                ]).wait()
                #expect(snapshots == [expected])
            }
        } catch {
            await device.close()
            throw error
        }
        await device.close()
    }
    #endif
}
