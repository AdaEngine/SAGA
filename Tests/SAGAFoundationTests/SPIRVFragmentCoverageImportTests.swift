import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func fragmentCoverageModule(
        samples: Bool = true, sampleCapability: Bool = true, maskWords: UInt32 = 1,
        depth: Bool = true, depthMode: Bool = true, compute: Bool = false
    ) -> [UInt32] {
        var words = head + i(17, 1)
        if samples && sampleCapability { words += i(17, 35) } // SampleRateShading
        words += i(14, 0, 1)
        var entry: [UInt32] = [compute ? 5 : 4, 30, 0x6e69616d, 0, 21, 22]
        if samples { entry.append(20) }
        if depth { entry.append(23) }
        words += [UInt32(entry.count + 1) << 16 | 15] + entry
        words += compute ? i(16, 30, 17, 1, 1, 1) : i(16, 30, 7)
        if depthMode { words += i(16, 30, 12) } // DepthReplacing
        if samples { words += i(71, 20, 11, 18) + i(71, 20, 14) }
        words += i(71, 21, 11, 20) + i(71, 21, 14) + i(71, 22, 11, 20)
        if depth { words += i(71, 23, 11, 22) }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(22, 4, 32),
            i(43, 2, 5, maskWords), i(43, 2, 6, 0), i(43, 4, 7, 0x3e800000), i(28, 8, 3, 5),
            i(32, 9, 1, 3), i(32, 10, 1, 8), i(32, 11, 3, 8), i(32, 12, 3, 3), i(32, 13, 3, 4), i(33, 14, 1),
        ] { words += instruction }
        if samples { words += i(59, 9, 20, 1) }
        words += i(59, 10, 21, 1) + i(59, 11, 22, 3)
        if depth { words += i(59, 13, 23, 3) }
        words += i(54, 1, 30, 0, 14) + i(248, 31)
        if samples { words += i(61, 3, 32, 20) }
        words += i(65, 9, 33, 21, 6) + i(61, 3, 34, 33) + i(65, 12, 35, 22, 6) + i(62, 35, 34)
        if depth { words += i(62, 23, 7) }
        return words + i(253) + i(56)
    }

    @Test func spirvFragmentCoverageNormalizesSignedInputsAndMaskArray() throws {
        let module = try parse(Self.fragmentCoverageModule())
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        #expect(entry.arguments.map(\.binding) == [.builtIn(.sampleMask), .builtIn(.sampleIndex)])
        for input in entry.arguments {
            #expect(try module.types.value(for: input.type).kind == .scalar(.u32))
        }
        let result = try #require(entry.result)
        guard case .structure(let outputs, _) = try module.types.value(for: result.type).kind else {
            throw ShaderFailure.invalidHandle
        }
        #expect(outputs.map(\.binding) == [.builtIn(.sampleMask), .builtIn(.fragDepth)])
        #expect(try outputs.map { try module.types.value(for: $0.type).kind } == [.scalar(.u32), .scalar(.f32)])
        #expect(entry.expressions.values.count(where: { if case .bitcast = $0.kind { true } else { false } }) == 3)
    }

    @Test func spirvFragmentSampleMaskDoesNotRequireSampleRateShading() throws {
        let words = Self.fragmentCoverageModule(samples: false, depth: false, depthMode: false)
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(words)).module
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        #expect(entry.arguments.first?.binding == .builtIn(.sampleMask))
        #expect(entry.result?.binding == .builtIn(.sampleMask))
    }

    @Test func spirvFragmentCoverageRejectsInvalidModeCapabilityAndShape() throws {
        try fails(Self.fragmentCoverageModule(sampleCapability: false), code: .unsupportedCapability)
        try fails(Self.fragmentCoverageModule(maskWords: 2), code: .unsupportedConstruct)
        try fails(Self.fragmentCoverageModule(depthMode: false))
        try fails(Self.fragmentCoverageModule(depth: false))
        try fails(Self.fragmentCoverageModule(compute: true))
    }

    @Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
    func spirvFragmentCoverageReachesTextBackends(format: ShaderOutputFormat) throws {
        let compiler = IndependentShaderCompiler()
        let shader = try compiler.compileRenderStage(.spirv(Self.fragmentCoverageModule()), entryPoint: "main", to: format)
        #expect(shader.writesSampleMask && shader.writesDepth)
        if format == .wgsl {
            guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
            _ = try compiler.parseAndValidate(.wgsl(source))
        }
    }
}
