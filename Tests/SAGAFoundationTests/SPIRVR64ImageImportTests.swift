import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    static func r64StorageImageModule(
        atomics: Bool = true, imageCapability: Bool = true, imageExtension: Bool = true,
        atomicCapability: Bool = true, atomicOpcode: UInt32 = 237
    ) -> [UInt32] {
        var words = head + i(17, 1)
        if !imageCapability { words += i(17, 11) }
        if atomics && atomicCapability { words += i(17, 12) }
        if imageCapability { words += i(17, 5016) }
        if imageExtension {
            words += i(10, 0x5f565053, 0x5f545845, 0x64616873, 0x695f7265, 0x6567616d, 0x746e695f, 0x3436)
        }
        words += i(14, 0, 1) + i(15, 5, 30, 0x6e69616d, 0) + i(16, 30, 17, 1, 1, 1)
        words += i(71, 20, 34, 0) + i(71, 20, 33, 0) + i(71, 21, 34, 0) + i(71, 21, 33, 1) + i(71, 21, 25)
        for instruction in [
            i(19, 1), i(21, 2, 64, 0), i(21, 3, 32, 0), i(23, 4, 2, 4), i(25, 5, 2, 1, 0, 0, 0, 2, 40),
            i(32, 6, 0, 5), i(33, 8, 1), i(21, 9, 32, 1), i(23, 10, 9, 2),
            i(43, 3, 11, 0), i(43, 3, 12, 1), i(43, 9, 13, 0),
            i(43, 2, 14, 0x00000001, 0x90000000), i(43, 2, 15, 0x76543210, 0xfedcba98), i(44, 10, 17, 13, 13),
        ] { words += instruction }
        if atomics { words += i(32, 7, 11, 2) }
        words += i(59, 6, 20, 0) + i(59, 6, 21, 0) + i(54, 1, 30, 0, 8) + i(248, 31)
        if atomics {
            words += i(60, 7, 32, 20, 17, 11) + i(atomicOpcode, 2, 33, 32, 12, 11, 14) + i(239, 2, 34, 32, 12, 11, 15)
        }
        return words + i(61, 5, 36, 20) + i(98, 4, 37, 36, 17) + i(61, 5, 38, 21) + i(99, 38, 17, 37) + i(253) + i(56)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvR64ImageImportPreservesLoadStoreAndUnsignedExtrema(_ format: ShaderOutputFormat, _ atomics: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.r64StorageImageModule(atomics: atomics))
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.storageTextures.map(\.format) == [.r64Uint, .r64Uint])
    #expect(shader.storageTextures.map(\.access) == [atomics ? [.load, .store, .atomic] : [.load, .store], [.store]])
    guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    if atomics {
        #expect(text.contains("10376293541461622785"))
        #expect(text.contains("18364758544493064720"))
    }
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.shaderInt64, .textureInt64Atomic])
    }
}

@Test func spirvR64ImageImportRejectsMissingDeclarationsAndUnsupportedAtomics() throws {
    let compiler = IndependentShaderCompiler()
    for (words, message) in [
        (SPIRVFrontendTests.r64StorageImageModule(imageCapability: false), "Int64Image"),
        (SPIRVFrontendTests.r64StorageImageModule(imageExtension: false), "SPV_EXT_shader_image_int64"),
        (SPIRVFrontendTests.r64StorageImageModule(atomicCapability: false), "Int64Atomics"),
        (SPIRVFrontendTests.r64StorageImageModule(atomicOpcode: 234), "min and max"),
    ] {
        do {
            _ = try compiler.parseAndValidate(.spirv(words), capabilities: [.shaderInt64, .textureInt64Atomic])
            Issue.record("Accepted unsupported R64ui image: \(message)")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.contains { $0.message.contains(message) })
        }
    }
}
