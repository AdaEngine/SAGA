import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func escapingAccessPathModule(chained: Bool = false, invalidEscape: Bool = false) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 10, 0x6e69616d, 0), i(16, 10, 17, 1, 1, 1),
            i(71, 4, 6, 4), i(71, 42, 6, 8), i(71, 5, 2), i(72, 5, 0, 35, 0), i(72, 5, 1, 35, 4), i(71, 8, 34, 0), i(71, 8, 33, 0),
            i(19, 1), i(20, 2), i(21, 3, 32, 0), i(21, 40, 32, 1), i(43, 40, 41, 1), i(43, 3, 13, 2),
            i(28, 4, 3, 13), i(28, 42, 4, 13), i(30, 5, 3, 42),
            i(32, 6, 12, 5), i(32, 7, 12, 3), i(32, 9, 12, 4), i(32, 43, 12, 42), i(33, 15, 1),
            i(43, 3, 11, 0), i(43, 3, 12, 1), i(43, 3, 16, 99), i(59, 6, 8, 12),
            i(54, 1, 10, 0, 15), i(248, 20), i(65, 7, 21, 8, 11), i(249, 22),
            i(248, 22), i(245, 3, 30, 11, 20, 37, 23), i(61, 3, 31, 21),
            i(194, 3, 45, 31, 12), i(199, 3, 46, 31, 12),
        ] {
            words += instruction
        }
        let address = chained
            ? i(65, 9, 32, 8, 41, 45) + i(66, 7, 33, 32, 46) + i(65, 7, 34, 33)
            : i(65, 7, 33, 8, 12, 45, 46) + i(83, 7, 34, 33)
        if !invalidEscape { words += address }
        words += i(176, 2, 35, 30, 13) + i(246, 24, 23, 0) + i(250, 35, 23, 24) + i(248, 23)
        if invalidEscape { words += address }
        return words + i(128, 3, 36, 31, 12) + i(62, 21, 36) + i(128, 3, 37, 30, 12) + i(249, 22)
            + i(248, 24) + i(62, 21, 11) + i(62, 34, 16) + i(253) + i(56)
    }

    @Test func spirvAccessPathRejectsNonDominatingPointerEvenWithAReachableBase() throws {
        try fails(Self.escapingAccessPathModule(invalidEscape: true))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func spirvAccessPathTranslatesEscapingPointers(_ format: ShaderOutputFormat, _ chained: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.spirv(SPIRVFrontendTests.escapingAccessPathModule(chained: chained))
    let shader = try compiler.compileCompute(source, to: format)
    guard case .text(let actual, let text) = shader.artifact else { Issue.record("Expected shader text"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvAccessPathStoresAtTheCapturedIndexOnMetal(_ chained: Bool) async throws {
    let source = ShaderInput.spirv(SPIRVFrontendTests.escapingAccessPathModule(chained: chained))
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let buffer = try await device.createBuffer(.init(size: 20, usage: [.storage, .copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 20, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .write(buffer: buffer, offset: 0, words: [0, 10, 20, 30, 40]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): buffer], workgroups: .init(x: 1)),
            .copy(source: buffer, destination: readback, byteCount: 20),
            .read(buffer: readback, byteCount: 20),
        ]).wait()
        #expect(result == [[0, 10, 20, 99, 40]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
