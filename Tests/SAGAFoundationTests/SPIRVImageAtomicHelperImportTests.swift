import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    static func imageAtomicHelperModule(
        descriptorArray: Bool = false, includePortable: Bool = false, atomicReadOnly: Bool = false,
        nonUniformID: UInt32? = nil
    ) -> [UInt32] {
        var words = head + i(17, 1)
        if nonUniformID != nil {
            words += i(17, 5301) + i(10, 0x5f565053, 0x5f545845, 0x63736564, 0x74706972, 0x695f726f, 0x7865646e, 0x676e69)
        }
        words += i(14, 0, 1) + i(15, 5, 30, 0x6d6f7461, 0x00006369)
        if includePortable { words += i(15, 5, 70, 0x74726f70, 0x656c6261, 0) }
        words += i(16, 30, 17, 1, 1, 1)
        if includePortable { words += i(16, 70, 17, 1, 1, 1) + i(5, 60, 0x6964726f, 0x7972616e, 0) }
        words += i(71, 20, 34, 0) + i(71, 20, 33, 0)
        if let nonUniformID { words += i(71, nonUniformID, 5300) }
        if atomicReadOnly { words += i(71, 20, 24) }
        if includePortable { words += i(71, 21, 34, 0) + i(71, 21, 33, 1) }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(23, 4, 3, 2), i(23, 18, 2, 4),
            i(25, 5, 2, 1, 0, 0, 0, 2, 33), i(32, 6, 0, 5), i(32, 7, 11, 2), i(33, 8, 1), i(33, 9, 1, 6, 4),
            i(43, 2, 10, 0), i(43, 2, 11, 1), i(43, 2, 12, 7), i(43, 3, 13, 0), i(44, 4, 14, 13, 13),
        ] { words += instruction }
        if descriptorArray { words += i(43, 2, 15, 2) + i(28, 16, 5, 15) + i(32, 17, 0, 16) }
        words += i(59, descriptorArray ? 17 : 6, 20, 0)
        if includePortable { words += i(59, 6, 21, 0) }
        for instruction in [
            i(54, 1, 40, 0, 9), i(55, 6, 41), i(55, 4, 42), i(248, 43),
            i(60, 7, 44, 41, 42, 10), i(234, 2, 45, 44, 11, 10, 12), i(253), i(56),
            i(54, 1, 50, 0, 9), i(55, 6, 51), i(55, 4, 52), i(248, 53), i(57, 1, 54, 40, 51, 52), i(253), i(56),
        ] { words += instruction }
        if includePortable {
            for instruction in [
                i(54, 1, 60, 0, 9), i(55, 6, 61), i(55, 4, 62), i(248, 63), i(61, 5, 64, 61),
                i(98, 18, 65, 64, 62), i(99, 64, 62, 65), i(253), i(56),
            ] { words += instruction }
        }
        words += i(54, 1, 30, 0, 8) + i(248, 31)
        if descriptorArray { words += i(65, 6, 32, 20, 11) }
        let image: UInt32 = descriptorArray ? 32 : 20
        words += i(57, 1, 34, 50, image, 14)
        if includePortable { words += i(57, 1, 35, 60, image, 14) }
        words += i(253) + i(56)
        if includePortable { words += i(54, 1, 70, 0, 8) + i(248, 71) + i(57, 1, 72, 60, 21, 14) + i(253) + i(56) }
        return words
    }
}

@Test func spirvImageAtomicHelperImportAcceptsNonUniformMetadataWithoutDynamicDescriptorSelection() throws {
    let compiler = IndependentShaderCompiler()
    for (descriptorArray, decorated) in [(false, UInt32(20)), (false, 41), (false, 51), (false, 44), (true, 32), (true, 44)] {
        let words = SPIRVFrontendTests.imageAtomicHelperModule(descriptorArray: descriptorArray, nonUniformID: decorated)
        let shader = try compiler.compileCompute(.spirv(words), to: .wgsl)
        #expect(shader.storageTextures.map(\.access) == [[.load, .store, .atomic]])
    }
}

@Test(arguments: [false, true])
func spirvImageAtomicHelperImportQualifiesTransitivePointerFormals(_ descriptorArray: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.imageAtomicHelperModule(descriptorArray: descriptorArray))
    let shader = try compiler.compileCompute(input, to: .wgsl)
    #expect(shader.storageTextures.map(\.access) == [[.load, .store, .atomic]])
    guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("textureAtomicAdd"))
    let capabilities = ShaderCapabilities.textureAtomic.union(.storageTextureBindingArray)
    _ = try compiler.parseAndValidate(.wgsl(text), capabilities: capabilities)
}

@Test func spirvImageAtomicHelperImportKeepsSharedOrdinaryFormalReadWrite() throws {
    let compiler = IndependentShaderCompiler()
    let words = SPIRVFrontendTests.imageAtomicHelperModule(includePortable: true)
    let module = try compiler.parseAndValidate(.spirv(words), capabilities: [.textureAtomic]).module
    let ordinary = try #require(module.functions.values.first { $0.name == "ordinary" })
    let formal = try module.types.value(for: ordinary.arguments[0].type)
    #expect(formal.kind.storageTexture?.access == [.load, .store])
    let portable = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "portable")
    #expect(portable.storageTextures.map(\.access) == [[.load, .store]])
}

@Test func spirvImageAtomicHelperImportRejectsReadOnlyActual() throws {
    let words = SPIRVFrontendTests.imageAtomicHelperModule(atomicReadOnly: true)
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.spirv(words), capabilities: [.textureAtomic])
        Issue.record("Accepted a read-only actual image for an atomic helper")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.message.contains("read-write") })
    }
}
