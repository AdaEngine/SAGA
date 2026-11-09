import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func demoteModule(
        version: UInt32 = 0x00010300, capability: Bool = true, extensionName: Bool = true,
        compute: Bool = false, demote: Bool = true, extraOperand: Bool = false
    ) -> [UInt32] {
        var words = head
        words[1] = version
        words += i(17, 1)
        if capability { words += i(17, 5379) }
        if extensionName {
            // SPV_EXT_demote_to_helper_invocation
            words += i(10, 0x5f565053, 0x5f545845, 0x6f6d6564, 0x745f6574, 0x65685f6f,
                       0x7265706c, 0x766e695f, 0x7461636f, 0x006e6f69)
        }
        words += i(14, 0, 1) + i(15, compute ? 5 : 4, 10, 0x6e69616d, 0)
        words += compute ? i(16, 10, 17, 1, 1, 1) : i(16, 10, 7)
        words += i(19, 1) + i(22, 2, 32) + i(33, 3, 1) + i(43, 2, 4, 0x3f800000)
        words += i(54, 1, 10, 0, 3) + i(248, 11)
        if demote {
            words += extraOperand ? i(5380, 4) : i(5380)
            if !compute { words += i(207, 2, 12, 4) }
            words += i(253)
        } else {
            words += i(252)
        }
        return words + i(56)
    }

    @Test(arguments: [false, true])
    func spirvDemotionKeepsSubsequentDerivativeAndReturn(core: Bool) throws {
        let words = Self.demoteModule(version: core ? 0x00010600 : 0x00010300, extensionName: !core)
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(words)).module
        let function = try module.functions.value(for: module.entryPoints[0].function)
        let handles = try function.blocks.value(for: function.body).statements
        let statements = try handles.map { try function.statements.value(for: $0).kind }
        try #require(statements.count == 3)
        guard case .demoteToHelperInvocation = statements[0], case .emit(let expressions) = statements[1],
              case .returnValue(nil) = statements[2] else {
            Issue.record("Expected demotion, derivative evaluation, and return in the same block")
            return
        }
        let derivative = try #require(expressions.first)
        guard case .derivative(.x, .none, _) = try function.expressions.value(for: derivative).kind else {
            Issue.record("Expected the derivative following demotion")
            return
        }
    }

    @Test func spirvKillRemainsATerminator() throws {
        let module = try parse(Self.demoteModule(capability: false, extensionName: false, demote: false))
        let function = try module.functions.value(for: module.entryPoints[0].function)
        let handles = try function.blocks.value(for: function.body).statements
        try #require(handles.count == 1)
        guard case .kill = try function.statements.value(for: handles[0]).kind else {
            Issue.record("Expected a terminating kill")
            return
        }
    }

    @Test func spirvDemotionRequiresCapabilityAndVersionOrExtension() throws {
        try fails(Self.demoteModule(capability: false), code: .unsupportedCapability)
        try fails(Self.demoteModule(extensionName: false), code: .unsupportedCapability)
        try fails(Self.demoteModule(extraOperand: true))
    }

    @Test func spirvDemotionRejectsComputeEntryPoint() throws {
        let words = Self.demoteModule(compute: true)
        do {
            _ = try IndependentShaderCompiler().parseAndValidate(.spirv(words))
            Issue.record("Accepted demotion in a compute entry point")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.first?.code == .invalidControlFlow)
        }
    }
}
