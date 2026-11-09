import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func conservativeDepthModule(
        modes: [UInt32], writesDepth: Bool = true, replacesDepth: Bool = true, compute: Bool = false,
        declaresUnusedDepth: Bool = false
    ) -> [UInt32] {
        var words = pre
        let model: UInt32 = compute ? 5 : 4
        if writesDepth {
            words += i(15, model, 20, 0x6e69616d, 0, 12, 13)
        } else if declaresUnusedDepth {
            words += i(15, model, 20, 0x6e69616d, 0, 13)
        } else {
            words += i(15, model, 20, 0x6e69616d, 0)
        }
        words += compute ? i(16, 20, 17, 1, 1, 1) : i(16, 20, 7)
        if replacesDepth { words += i(16, 20, 12) }
        for mode in modes { words += i(16, 20, mode) }
        if writesDepth { words += i(71, 12, 11, 15) }
        if writesDepth || declaresUnusedDepth { words += i(71, 13, 11, 22) }
        words += i(19, 1) + i(33, 2, 1)
        if writesDepth || declaresUnusedDepth {
            words += i(22, 3, 32) + i(23, 4, 3, 4) + i(32, 5, 1, 4) + i(32, 6, 3, 3)
            if writesDepth { words += i(59, 5, 12, 1) }
            words += i(59, 6, 13, 3)
        }
        words += i(54, 1, 20, 0, 2) + i(248, 21)
        if writesDepth { words += i(61, 4, 22, 12) + i(81, 3, 23, 22, 2) + i(62, 13, 23) }
        return words + i(253) + i(56)
    }

    @Test(arguments: [IRConservativeDepth.greaterEqual, .lessEqual, .unchanged])
    func spirvConservativeDepthRetainsEachHint(_ depth: IRConservativeDepth) throws {
        let mode: UInt32 = switch depth {
        case .greaterEqual: 14
        case .lessEqual: 15
        case .unchanged: 16
        }
        let compiler = IndependentShaderCompiler()
        let words = Self.conservativeDepthModule(modes: [mode])
        let module = try compiler.parseAndValidate(.spirv(words), capabilities: [.earlyDepthTest]).module
        let entry = module.entryPoints[0]
        #expect(entry.earlyDepthTest == .allow(depth))
        let function = try module.functions.value(for: entry.function)
        #expect(function.result?.binding == .builtIn(.fragDepth))
        let artifact = try compiler.compileRenderStage(.spirv(words), entryPoint: "main", to: .wgsl)
        #expect(artifact.earlyDepthTest == .allow(depth))
        guard case .text(.wgsl, let source) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.earlyDepthTest])
    }

    @Test(arguments: [false, true])
    func spirvConservativeDepthPermitsHintWithoutDepthOutput(declaresUnusedDepth: Bool) throws {
        let words = Self.conservativeDepthModule(modes: [14], writesDepth: false, replacesDepth: false,
                                                declaresUnusedDepth: declaresUnusedDepth)
        let module = try parse(words)
        let entry = module.entryPoints[0]
        #expect(entry.earlyDepthTest == .allow(.greaterEqual))
        let function = try module.functions.value(for: entry.function)
        #expect(function.result == nil)
    }

    @Test(arguments: [[UInt32(9), 14], [14, 9]])
    func spirvForcedEarlyDepthTakesPrecedence(_ modes: [UInt32]) throws {
        let module = try parse(Self.conservativeDepthModule(modes: modes))
        #expect(module.entryPoints[0].earlyDepthTest == .force)
    }

    @Test func spirvConservativeDepthRejectsConflictingAndIncompatibleModes() throws {
        try fails(Self.conservativeDepthModule(modes: [14, 15]))
        try fails(Self.conservativeDepthModule(modes: [9, 14, 16]))
        try fails(Self.conservativeDepthModule(modes: [14], replacesDepth: false))
        try fails(Self.conservativeDepthModule(modes: [14], writesDepth: false, replacesDepth: false, compute: true))
    }
}
