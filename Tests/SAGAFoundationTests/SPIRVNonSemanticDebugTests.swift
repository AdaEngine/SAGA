import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    enum NonSemanticDebugFailure: CaseIterable, Sendable {
        case missingExtension, unknownSet, resultType, forwardOperand, undefinedOperand
        case duplicateResult, semanticOperand, semanticDecoration, debugBeforePhi

        var diagnostic: ShaderDiagnosticCode {
            switch self {
            case .missingExtension: .unsupportedCapability
            case .unknownSet: .unsupportedConstruct
            default: .invalidBinary
            }
        }
    }

    static func nonSemanticDebugModule(core: Bool = false, failure: NonSemanticDebugFailure? = nil) -> [UInt32] {
        func string(_ opcode: UInt32, prefix: [UInt32] = [], _ text: String) -> [UInt32] {
            var bytes = Array(text.utf8) + [0]
            while bytes.count % 4 != 0 { bytes.append(0) }
            let encoded = stride(from: 0, to: bytes.count, by: 4).map { offset in
                (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << ($1 * 8) }
            }
            return [UInt32(1 + prefix.count + encoded.count) << 16 | opcode] + prefix + encoded
        }
        var words = head + i(17, 1)
        if core { words[1] = 0x00010600 }
        if !core && failure != .missingExtension { words += string(10, "SPV_KHR_non_semantic_info") }
        let set = failure == .unknownSet ? "NonSemantic.Unknown.100" : "NonSemantic.Shader.DebugInfo.100"
        words += string(11, prefix: [30], set) + i(14, 0, 1)
        words += core ? i(15, 5, 20, 0x6e69616d, 0, 10) : i(15, 5, 20, 0x6e69616d, 0)
        words += i(16, 20, 17, 1, 1, 1) + string(7, prefix: [31], "unit.wgsl")
        words += i(71, 3, 2) + i(72, 3, 0, 35, 0) + i(71, 10, 34, 0) + i(71, 10, 33, 0)
        if failure == .semanticDecoration { words += i(71, 32, 24) }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(30, 3, 2), i(32, 4, 12, 3), i(32, 5, 12, 2), i(33, 6, 1), i(20, 7),
            i(43, 2, 11, 0), i(43, 2, 12, 42), i(43, 2, 13, 1), i(41, 7, 14), i(59, 4, 10, 12),
            i(12, failure == .resultType ? 2 : 1, failure == .duplicateResult ? 12 : 32, 30, 35,
              failure == .forwardOperand ? 33 : failure == .undefinedOperand ? 99 : 31), // DebugSource
            i(54, 1, 20, 0, 6), i(248, 21),
        ] { words += instruction }
        if failure == .debugBeforePhi {
            for instruction in [
                i(247, 24, 0), i(250, 14, 25, 26), i(248, 25), i(249, 24), i(248, 26), i(249, 24), i(248, 24),
                i(12, 1, 33, 30, 24), i(245, 2, 27, 12, 25, 13, 26), i(253), i(56),
            ] { words += instruction }
            return words
        }
        for instruction in [
            i(12, 1, 33, 30, 103, 32, 13, 13, 13, 13), // DebugLine uses the global DebugSource result.
            i(65, 5, 22, 10, 11), i(62, 22, failure == .semanticOperand ? 33 : 12), i(253),
            i(12, 1, 34, 30, 24), i(56), // DebugNoScope may appear after the terminator.
        ] { words += instruction }
        return words
    }

    @Test(arguments: NonSemanticDebugFailure.allCases)
    func spirvNonSemanticDebugRejectsInvalidIDsAndSemanticUses(_ failure: NonSemanticDebugFailure) throws {
        try fails(Self.nonSemanticDebugModule(failure: failure), code: failure.diagnostic)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvNonSemanticDebugPreservesSemanticStorageWrites(_ format: ShaderOutputFormat, _ core: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.nonSemanticDebugModule(core: core))
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [4])
    guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("42u") && !text.contains("unit.wgsl"))
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}
