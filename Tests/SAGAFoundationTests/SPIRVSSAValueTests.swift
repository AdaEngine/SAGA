import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func escapingLoopValueModule(invalidEscape: Bool = false) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 10, 0x6e69616d, 0), i(16, 10, 17, 1, 1, 1),
            i(71, 4, 2), i(72, 4, 0, 35, 0), i(72, 4, 1, 35, 4), i(72, 4, 2, 35, 8), i(71, 8, 34, 0), i(71, 8, 33, 0),
            i(19, 1), i(20, 2), i(21, 3, 32, 0), i(30, 4, 3, 3, 3), i(32, 5, 12, 4), i(32, 6, 12, 3), i(33, 9, 1),
            i(43, 3, 11, 0), i(43, 3, 12, 1), i(43, 3, 13, 2), i(43, 3, 14, 3), i(43, 3, 15, 10), i(43, 3, 16, 99),
            i(59, 5, 8, 12), i(33, 40, 3, 3, 3),
            i(54, 3, 41, 0, 40), i(55, 3, 42), i(55, 3, 43), i(248, 44), i(128, 3, 45, 42, 43), i(254, 45), i(56),
            i(54, 1, 10, 0, 9), i(248, 20),
            i(65, 6, 21, 8, 11), i(65, 6, 22, 8, 12), i(65, 6, 23, 8, 13), i(249, 24),
            i(248, 24), i(245, 3, 30, 11, 20, 35, 25), i(61, 3, 31, 21), i(83, 3, 32, 31),
        ] {
            words += instruction
        }
        if !invalidEscape { words += i(57, 3, 33, 41, 32, 30) }
        words += i(176, 2, 34, 30, 14) + i(246, 26, 25, 0) + i(250, 34, 25, 26) + i(248, 25)
        if invalidEscape { words += i(57, 3, 33, 41, 32, 30) }
        return words + i(128, 3, 36, 33, 15) + i(62, 21, 36) + i(128, 3, 35, 30, 12) + i(249, 24)
            + i(248, 26) + i(62, 21, 16) + i(62, 22, 32) + i(62, 23, 33) + i(253) + i(56)
    }

    @Test func spirvSSAValueRejectsAnEscapeThatDoesNotDominateItsUse() throws {
        try fails(Self.escapingLoopValueModule(invalidEscape: true))
    }

    @Test func spirvSSAValueUsesThePhiPredecessorForDominance() throws {
        var words = Self.pre
        for instruction in [
            Self.i(20, 1), Self.i(21, 2, 32, 0), Self.i(43, 2, 3, 0), Self.i(43, 2, 4, 1),
            Self.i(43, 2, 5, 3), Self.i(33, 6, 2, 1), Self.i(54, 2, 10, 0, 6), Self.i(55, 1, 11),
            Self.i(248, 20), Self.i(247, 27, 0), Self.i(250, 11, 21, 26),
            Self.i(248, 21), Self.i(249, 22),
            Self.i(248, 22), Self.i(245, 2, 30, 3, 21, 31, 23), Self.i(128, 2, 31, 30, 4),
            Self.i(176, 1, 32, 30, 5), Self.i(246, 24, 23, 0), Self.i(250, 32, 23, 24),
            Self.i(248, 23), Self.i(249, 22), Self.i(248, 24), Self.i(249, 27), Self.i(248, 26), Self.i(249, 27),
            Self.i(248, 27), Self.i(245, 2, 33, 31, 24, 3, 26), Self.i(254, 33), Self.i(56),
        ] {
            words += instruction
        }
        let function = try IndependentShaderCompiler().parseAndValidate(.spirv(words)).module.functions.values[0]
        #expect(function.statements.values.filter { if case .store = $0.kind { true } else { false } }.count == 5)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvSSAValueTranslatesLoopHeaderSnapshots(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let result = try compiler.compileCompute(.spirv(SPIRVFrontendTests.escapingLoopValueModule()), to: format)
    guard case .text(let actual, let text) = result.artifact else { Issue.record("Expected shader text"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
    if format == .glsl { _ = try compiler.parseAndValidate(.glsl(text, stage: .compute, defines: [:])) }
}

#if canImport(Metal)
@Test func spirvSSAValueReadsTheLastHeaderSnapshotOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.escapingLoopValueModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let buffer = try await device.createBuffer(.init(size: 12, usage: [.storage, .copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 12, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .write(buffer: buffer, offset: 0, words: [7, 0, 0]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): buffer], workgroups: .init(x: 1)),
            .copy(source: buffer, destination: readback, byteCount: 12),
            .read(buffer: readback, byteCount: 12),
        ]).wait()
        #expect(result == [[99, 40, 43]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
