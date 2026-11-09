import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func dualSourceModule(
        outputs: [(location: UInt32, index: UInt32?, width: UInt32)],
        model: UInt32 = 4, extraDecorations: [[UInt32]] = []
    ) -> [UInt32] {
        var words = pre
        let interfaces = outputs.indices.map { 20 + UInt32($0) }
        words += [UInt32(interfaces.count + 5) << 16 | 15, model, 30, 0x6e69616d, 0] + interfaces
        if model == 4 { words += i(16, 30, 7) }
        for (id, output) in zip(interfaces, outputs) {
            words += i(71, id, 30, output.location)
            if let index = output.index { words += i(71, id, 32, index) }
        }
        for decoration in extraDecorations { words += decoration }
        for instruction in [
            i(19, 1), i(22, 2, 32), i(23, 3, 2, 4), i(23, 4, 2, 3),
            i(32, 5, 3, 3), i(32, 6, 3, 4), i(33, 7, 1),
            i(43, 2, 8, 0x3f000000), i(44, 3, 9, 8, 8, 8, 8), i(44, 4, 10, 8, 8, 8),
        ] { words += instruction }
        for (id, output) in zip(interfaces, outputs) { words += i(59, output.width == 4 ? 5 : 6, id, 3) }
        words += i(54, 1, 30, 0, 7) + i(248, 31)
        for (id, output) in zip(interfaces, outputs) { words += i(62, id, output.width == 4 ? 9 : 10) }
        return words + i(253) + i(56)
    }

    @Test(arguments: [false, true])
    func spirvDualSourceNormalizesExplicitAndImplicitPrimary(explicitPrimary: Bool) throws {
        let words = Self.dualSourceModule(outputs: [(0, 1, 4), (0, explicitPrimary ? 0 : nil, 4)])
        let module = try parse(words)
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        let result = try #require(entry.result)
        guard case .structure(let members, _) = try module.types.value(for: result.type).kind else {
            throw ShaderFailure.invalidHandle
        }
        #expect(members.map(\.binding) == [
            .location(index: 0, interpolation: nil, sampling: nil, blendSource: 1, perPrimitive: false),
            .location(index: 0, interpolation: nil, sampling: nil, blendSource: 0, perPrimitive: false),
        ])
        #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.spirv(words)) }
        _ = try IndependentShaderCompiler().parseAndValidate(.spirv(words), capabilities: [.dualSourceBlending])
    }

    @Test func spirvExplicitZeroIndexWithoutSecondaryRemainsOrdinaryColor() throws {
        let module = try parse(Self.dualSourceModule(outputs: [(1, 0, 4)]))
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        #expect(entry.result?.binding == .location(index: 1, interpolation: nil, sampling: nil,
                                                    blendSource: nil, perPrimitive: false))
    }

    @Test func spirvDualSourceRejectsInvalidIndicesAndUnsupportedPairs() throws {
        try fails(Self.dualSourceModule(outputs: [(0, nil, 4), (0, 0, 4)]))
        try fails(Self.dualSourceModule(outputs: [(0, 1, 4), (0, 1, 4)]))
        try fails(Self.dualSourceModule(outputs: [(0, 2, 4)]))
        try fails(Self.dualSourceModule(outputs: [(0, 0, 4)], model: 0))
        try fails(Self.dualSourceModule(outputs: [(0, 0, 4)], extraDecorations: [Self.i(71, 20, 32, 0)]))
        try fails(Self.dualSourceModule(outputs: [(0, 1, 4)]), code: .unsupportedConstruct)
        try fails(Self.dualSourceModule(outputs: [(1, 0, 4), (1, 1, 4)]), code: .unsupportedConstruct)
        try fails(Self.dualSourceModule(outputs: [(0, 0, 4), (0, 1, 3)]), code: .unsupportedConstruct)
        try fails(Self.dualSourceModule(outputs: [(0, 0, 4), (0, 1, 4), (1, nil, 4)]), code: .unsupportedConstruct)
    }

    @Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
    func spirvDualSourceReachesTextBackends(format: ShaderOutputFormat) throws {
        let compiler = IndependentShaderCompiler()
        let words = Self.dualSourceModule(outputs: [(0, nil, 4), (0, 1, 4)])
        let stage = try compiler.compileRenderStage(.spirv(words), entryPoint: "main", to: format)
        #expect(stage.usesDualSourceBlending)
        #expect(stage.colorOutputKinds == [0: .float])
        if format == .wgsl {
            guard case .text(.wgsl, let source) = stage.artifact else { throw ShaderFailure.incompatibleArtifact }
            _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.dualSourceBlending])
        }
    }
}
