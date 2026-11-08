import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func atomicCompareExchangeModule(unequalSemantics: UInt32 = 0) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 10, 0x6e69616d, 0, 14), i(16, 10, 17, 1, 1, 1),
            i(71, 14, 11, 28), i(71, 3, 6, 4), i(71, 4, 2), i(72, 4, 0, 35, 0),
            i(71, 5, 2), i(72, 5, 0, 35, 0), i(72, 5, 1, 35, 4), i(72, 5, 2, 35, 8), i(72, 5, 3, 35, 12),
            i(71, 11, 34, 0), i(71, 11, 33, 0), i(71, 12, 34, 0), i(71, 12, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(43, 2, 20, 0), i(43, 2, 21, 1), i(43, 2, 22, 2), i(43, 2, 23, 3),
            i(43, 2, 24, 7), i(43, 2, 25, 99), i(43, 2, 26, 11), i(43, 2, 27, unequalSemantics),
            i(28, 3, 2, 21), i(30, 4, 3), i(30, 5, 2, 2, 2, 2), i(32, 6, 12, 4), i(32, 7, 12, 5),
            i(32, 8, 12, 2), i(33, 9, 1), i(23, 15, 2, 3), i(32, 16, 1, 15), i(32, 17, 1, 2),
            i(59, 6, 11, 12), i(59, 7, 12, 12), i(59, 16, 14, 1),
            i(54, 1, 10, 0, 9), i(248, 13),
            i(65, 8, 30, 11, 20, 20),
            i(230, 2, 31, 30, 21, 20, 27, 24, 20), // Successful exchange returns 0.
            i(230, 2, 32, 30, 21, 20, 27, 26, 25), // Mismatch returns 7.
            i(65, 17, 33, 14, 20), i(61, 2, 34, 33), i(128, 2, 35, 34, 21),
            i(65, 8, 36, 11, 20, 35),
            i(230, 2, 37, 36, 21, 20, 27, 25, 20), // Robust out-of-bounds exchange returns 0.
            i(227, 2, 38, 30, 21, 20),
            i(65, 8, 40, 12, 20), i(62, 40, 31), i(65, 8, 41, 12, 21), i(62, 41, 32),
            i(65, 8, 42, 12, 22), i(62, 42, 38), i(65, 8, 43, 12, 23), i(62, 43, 37),
            i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvCompareExchangeRejectsUnequalAcquireSemantics() throws {
        try fails(Self.atomicCompareExchangeModule(unequalSemantics: 0x42), code: .unsupportedConstruct)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvCompareExchangeTranslates(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.atomicCompareExchangeModule()), to: format)
    guard case .text(let actual, let code) = shader.artifact else { Issue.record("Expected text shader"); return }
    #expect(actual == format)
    switch format {
    case .wgsl:
        #expect(code.contains("atomicCompareExchangeWeak("))
        #expect(code.contains("loop {"))
        _ = try compiler.parseAndValidate(.wgsl(code))
    case .msl:
        #expect(code.contains("atomic_compare_exchange_weak_explicit("))
        #expect(code.contains("while ("))
    case .hlsl: #expect(code.contains("InterlockedCompareExchange("))
    case .glsl: #expect(code.contains("atomicCompSwap("))
    default: Issue.record("Unexpected shader format")
    }
}

#if canImport(Metal)
@Test func spirvCompareExchangeReturnsPreviousValuesOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.atomicCompareExchangeModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let counter = try await device.createBuffer(.init(size: 4, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .write(buffer: counter, offset: 0, words: [0]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): counter, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(result == [[0, 7, 7, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
