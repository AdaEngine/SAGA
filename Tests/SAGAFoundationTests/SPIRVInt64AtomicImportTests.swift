import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func int64AtomicModule(
        capability: Bool = true, metalMinMaxOnly: Bool = false, oppositeOrder: Bool = false
    ) -> [UInt32] {
        var words = head + i(17, 1) + i(17, 11)
        if capability { words += i(17, 12) } // Int64Atomics
        for instruction in [
            i(14, 0, 1), i(15, 5, 20, 0x6e69616d, 0), i(16, 20, 17, 1, 1, 1),
            i(71, 5, 2), i(72, 5, 0, 35, 0), i(72, 5, 1, 35, 8),
            i(71, 17, 34, 0), i(71, 17, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 64, 1), i(21, 4, 64, 0),
            i(30, 5, 3, 4), i(32, 6, 12, 5), i(32, 7, 12, 3), i(32, 8, 12, 4), i(33, 9, 1),
            i(43, 2, 10, 0), i(43, 2, 11, 1),
            i(43, 3, 12, 0xfffffffd, 0xffffffff), i(43, 4, 13, 0, 0x80000000),
            i(59, 6, 17, 12), i(54, 1, 20, 0, 9), i(248, 21), i(65, 8, 23, 17, 11),
        ] { words += instruction }
        if !metalMinMaxOnly {
            for instruction in [
                i(65, 7, 22, 17, 10), i(228, 22, 11, 10, 12), i(227, 3, 24, 22, 11, 10),
                i(232, 3, 25, 22, 11, 10), i(233, 4, 26, 23, 11, 10),
                i(oppositeOrder ? 237 : 236, 3, 27, 22, 11, 10, 24), i(240, 4, 28, 23, 11, 10, 26),
                i(230, 3, 29, 22, 11, 10, 10, 24, 25), i(230, 4, 30, 23, 11, 10, 10, 13, 28),
                i(228, 22, 11, 10, 29), i(228, 23, 11, 10, 30),
            ] { words += instruction }
        }
        words += i(oppositeOrder ? 238 : 239, 4, 31, 23, 11, 10, 13) + i(253) + i(56)
        return words
    }

    @Test func spirvInt64AtomicCompareResultsUseWideLayout() throws {
        let module = try parse(Self.int64AtomicModule())
        for scalar: IRScalar in [.i64, .u64] {
            let compare = try #require(module.specialTypes.predeclared.first {
                $0.kind == .atomicCompareExchangeWeakResult(scalar)
            })
            guard case .structure(let members, let span) = try module.types.value(for: compare.type).kind else {
                throw ShaderFailure.invalidHandle
            }
            #expect(span == 16)
            #expect(members.map(\.offset) == [0, 8])
            #expect(try module.types.value(for: members[0].type).kind == .scalar(scalar))
        }
    }

    @Test func spirvInt64AtomicsRequireAtomicCapability() throws {
        try fails(Self.int64AtomicModule(capability: false, metalMinMaxOnly: true), code: .unsupportedCapability)
        try fails(Self.int64AtomicModule(capability: false), code: .unsupportedCapability)
    }

    @Test(arguments: [ShaderOutputFormat.wgsl, .hlsl, .glsl], [false, true])
    func spirvInt64AtomicsReachNativeTextTargets(format: ShaderOutputFormat, oppositeOrder: Bool) throws {
        let compiler = IndependentShaderCompiler()
        let shader = try compiler.compileCompute(.spirv(Self.int64AtomicModule(oppositeOrder: oppositeOrder)), to: format)
        #expect(shader.requiresShaderInt64AtomicAllOps)
        if format == .wgsl {
            guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
            _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .allKnown)
        }
    }

    @Test func spirvInt64AtomicMinMaxUseOpcodeSignedness() throws {
        let module = try parse(Self.int64AtomicModule(oppositeOrder: true))
        let operations = module.functions.values.flatMap { function in
            function.statements.values.compactMap { statement -> IRAtomicFunction? in
                guard case .atomic(_, let operation, _, _) = statement.kind else { return nil }
                return operation
            }
        }
        #expect(operations.contains(.minWithOppositeSignedness))
        #expect(operations.contains(.maxWithOppositeSignedness))
    }

    @Test func spirvUnusedUInt64MinMaxResultKeepsMetalSubset() throws {
        let compiler = IndependentShaderCompiler()
        let shader = try compiler.compileCompute(.spirv(Self.int64AtomicModule(metalMinMaxOnly: true)), to: .msl)
        #expect(shader.requiresShaderInt64AtomicMinMax)
        #expect(!shader.requiresShaderInt64AtomicAllOps)
    }
}
