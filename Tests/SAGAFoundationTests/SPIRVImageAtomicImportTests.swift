import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    static func imageAtomicModule(useResult: Bool = false, readOnly: Bool = false,
                                  signedOpcode: UInt32 = 236, semantics: UInt32 = 0) -> [UInt32] {
        var words = pre + i(15, 5, 30, 0x6e69616d, 0) + i(16, 30, 17, 1, 1, 1)
        words += i(71, 20, 34, 0) + i(71, 20, 33, 0) + i(71, 21, 34, 0) + i(71, 21, 33, 1)
        if readOnly { words += i(71, 20, 24) }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(23, 4, 3, 2), i(23, 5, 3, 3),
            i(25, 6, 2, 1, 0, 0, 0, 2, 33), i(25, 7, 3, 1, 0, 1, 0, 2, 24),
            i(32, 8, 0, 6), i(32, 9, 0, 7), i(32, 10, 11, 2), i(32, 11, 11, 3), i(33, 12, 1),
            i(43, 2, 13, 0), i(43, 2, 14, 1), i(43, 2, 15, 7), i(43, 3, 16, 0), i(43, 3, 17, 0xfffffffb),
            i(44, 4, 18, 16, 16), i(44, 5, 19, 16, 16, 16), i(43, 2, 22, semantics),
            i(59, 8, 20, 0), i(59, 9, 21, 0), i(54, 1, 30, 0, 12), i(248, 31),
            i(60, 10, 32, 20, 18, 13), i(234, 2, 33, 32, 14, 22, 15),
            i(60, 11, 34, 21, 19, 13), i(signedOpcode, 3, 35, 34, 14, 13, 17),
        ] { words += instruction }
        if useResult { words += i(128, 2, 36, 33, 15) }
        return words + i(253) + i(56)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvImageAtomicImportPreservesAccessAndSignedness(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.imageAtomicModule()), to: format)
    #expect(shader.storageTextures.map(\.access) == [[.load, .store, .atomic], [.load, .store, .atomic]])
    #expect(shader.storageTextures.map(\.format) == [.r32Uint, .r32Sint])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(text.contains("textureAtomicAdd") && text.contains("textureAtomicMin"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.textureAtomic])
    }
}

@Test func spirvImageAtomicImportRejectsUnsupportedResultUseAndOrdering() throws {
    let compiler = IndependentShaderCompiler()
    for (words, reason) in [
        (SPIRVFrontendTests.imageAtomicModule(useResult: true), "result"),
        (SPIRVFrontendTests.imageAtomicModule(readOnly: true), "read-write"),
        (SPIRVFrontendTests.imageAtomicModule(signedOpcode: 237), "signedness"),
        (SPIRVFrontendTests.imageAtomicModule(semantics: 0x802), "relaxed"),
    ] {
        do {
            _ = try compiler.parseAndValidate(.spirv(words), capabilities: [.textureAtomic])
            Issue.record("Accepted unsupported image atomic: \(reason)")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.contains { $0.message.contains(reason) })
        }
    }
}
