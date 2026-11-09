import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    static func matrixStorageLayoutModule(major: UInt32 = 5, stride: UInt32 = 16, decorateNested: Bool = true) -> [UInt32] {
        var words = pre + i(15, 5, 30, 0x6e69616d, 0) + i(16, 30, 17, 1, 1, 1)
        for instruction in [
            i(71, 6, 6, 32), i(71, 7, 6, 64), i(71, 8, 2), i(71, 9, 2),
            i(72, 8, 0, 35, 0), i(72, 8, 0, major), i(72, 8, 0, 7, stride), i(72, 8, 1, 35, 32),
            i(72, 9, 0, 35, 0), i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
        ] { words += instruction }
        if decorateNested { words += i(72, 8, 1, 5) + i(72, 8, 1, 7, 16) }
        for instruction in [
            i(19, 1), i(22, 2, 32), i(23, 3, 2, 3), i(24, 4, 3, 2), i(21, 5, 32, 0),
            i(43, 5, 10, 0), i(43, 5, 11, 1), i(43, 5, 12, 2), i(28, 6, 4, 12), i(28, 7, 6, 12),
            i(30, 8, 4, 7), i(30, 9, 2), i(32, 13, 12, 8), i(32, 14, 12, 9), i(32, 15, 12, 2),
            i(33, 16, 1), i(59, 13, 20, 12), i(59, 14, 21, 12), i(54, 1, 30, 0, 16), i(248, 31),
            i(65, 15, 32, 20, 10, 11, 12), i(61, 2, 33, 32),
            i(65, 15, 34, 20, 11, 11, 10, 11, 12), i(61, 2, 35, 34),
            i(129, 2, 36, 33, 35), i(65, 15, 37, 21, 10), i(62, 37, 36), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvMatrixStorageLayoutRequiresNaturalColumnMajor() throws {
        try fails(Self.matrixStorageLayoutModule(major: 4), code: .unsupportedConstruct)
        try fails(Self.matrixStorageLayoutModule(stride: 32), code: .unsupportedConstruct)
        try fails(Self.matrixStorageLayoutModule(decorateNested: false), code: .unsupportedConstruct)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvMatrixStorageLayoutPreservesNestedColumnMajorArrays(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.matrixStorageLayoutModule())
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [160, 4])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}
