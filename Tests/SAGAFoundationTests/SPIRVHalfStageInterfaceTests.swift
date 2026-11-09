import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    static func halfStageInterfaceModule(arithmetic: Bool, inputOutput: Bool = true) -> [UInt32] {
        var words = head + i(17, 1)
        if arithmetic { words += i(17, 9) }
        if inputOutput { words += i(17, 4436) }
        words += i(14, 0, 1) + i(15, 4, 20, 0x6e69616d, 0, 12, 13) + i(16, 20, 7)
        for instruction in [
            i(71, 12, 30, 0), i(71, 12, 13), i(71, 13, 30, 0),
            i(19, 1), i(22, 2, 16), i(23, 3, 2, 2), i(22, 4, 32), i(23, 5, 4, 2),
            i(32, 6, 1, 3), i(32, 7, 3, 5), i(33, 8, 1), i(59, 6, 12, 1), i(59, 7, 13, 3),
            i(54, 1, 20, 0, 8), i(248, 21), i(61, 3, 22, 12), i(115, 5, 23, 22),
            i(62, 13, 23), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvHalfInterfaceAndBufferCapabilitiesAreIndependent() throws {
        try fails(Self.halfStageInterfaceModule(arithmetic: true, inputOutput: false), code: .unsupportedCapability)
        var storage = Self.halfStorageModule(capability: false)
        storage.insert(contentsOf: Self.i(17, 4436), at: 7)
        try fails(storage, code: .unsupportedCapability)
    }
}

@Test(arguments: [true, false])
func spirvHalfInterfaceImportsLinearLocationsWithOrWithoutArithmetic(_ arithmetic: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.halfStageInterfaceModule(arithmetic: arithmetic))
    let shader = try compiler.compileRenderStage(input, entryPoint: "main", to: .wgsl)
    #expect(shader.varyings.first?.scalar == .f16)
    #expect(shader.varyings.first?.interpolation == .linear)
    #expect(shader.colorOutputComponentCounts == [0: 2])
    guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.shaderFloat16, .linearInterpolation])
}
