import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func derivativeModule(
        opcodes: [UInt32], controlCapability: Bool = true, compute: Bool = false,
        width: UInt32 = 32, mismatchedOperand: Bool = false
    ) -> [UInt32] {
        var words = head + i(17, 1)
        if width == 16 { words += i(17, 9) } // Float16
        if controlCapability { words += i(17, 51) } // DerivativeControl
        words += i(14, 0, 1) + i(15, compute ? 5 : 4, 30, 0x6e69616d, 0)
        words += compute ? i(16, 30, 17, 1, 1, 1) : i(16, 30, 7)
        for instruction in [
            i(19, 1), i(22, 2, width), i(23, 3, 2, 2), i(33, 4, 1), i(33, 5, 1, 2),
            i(43, 2, 6, width == 16 ? 0x3c00 : 0x3f800000),
            i(54, 1, 10, 0, 5), i(55, 2, 11), i(248, 12), i(80, 3, 13, 11, 11),
        ] { words += instruction }
        for (index, opcode) in opcodes.enumerated() {
            let vector = index.isMultiple(of: 2)
            let operand: UInt32 = vector && !mismatchedOperand ? 13 : 11
            words += i(opcode, vector ? 3 : 2, 14 + UInt32(index), operand)
        }
        for instruction in [
            i(253), i(56), i(54, 1, 30, 0, 4), i(248, 31), i(57, 1, 32, 10, 6), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvDerivativesPreserveAxisAndControl() throws {
        let module = try parse(Self.derivativeModule(opcodes: Array(207...215)))
        let axesAndControls = module.functions.values.flatMap { function in
            function.expressions.values.compactMap { expression -> String? in
                guard case .derivative(let axis, let control, _) = expression.kind else { return nil }
                return "\(axis.rawValue):\(control.rawValue)"
            }
        }
        #expect(axesAndControls == [
            "x:none", "y:none", "width:none",
            "x:fine", "y:fine", "width:fine",
            "x:coarse", "y:coarse", "width:coarse",
        ])
    }

    @Test func spirvDerivativeRequiresCapabilityOnlyForExplicitControl() throws {
        _ = try parse(Self.derivativeModule(opcodes: [207, 208, 209], controlCapability: false))
        try fails(Self.derivativeModule(opcodes: [210], controlCapability: false), code: .unsupportedCapability)
    }

    @Test func spirvDerivativeRejectsMismatchedAndHalfOperands() throws {
        try fails(Self.derivativeModule(opcodes: [207], mismatchedOperand: true))
        try fails(Self.derivativeModule(opcodes: [207], width: 16))
    }

    @Test func spirvDerivativeHelperRejectsComputeEntryPoint() throws {
        let shader = ShaderInput.spirv(Self.derivativeModule(opcodes: [207], compute: true))
        do {
            _ = try IndependentShaderCompiler().parseAndValidate(shader)
            Issue.record("Accepted a compute entry point calling a derivative")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.first?.code == .invalidControlFlow)
            #expect(diagnostics.first?.message.contains("incompatible with entrypoint stage") == true)
        }
    }
}
