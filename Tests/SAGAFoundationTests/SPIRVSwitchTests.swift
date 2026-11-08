import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func switchModule(
        labels: [UInt32] = [UInt32.max, 21, 0, 21, 2, 23, 9, 24], earlyBreak: Bool = false
    ) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 10, 0x6e69616d, 0), i(16, 10, 17, 1, 1, 1),
            i(71, 4, 2), i(72, 4, 0, 35, 0), i(72, 4, 1, 35, 4), i(71, 8, 34, 0), i(71, 8, 33, 0),
            i(19, 1), i(21, 2, 32, 1), i(21, 3, 32, 0), i(30, 4, 2, 2), i(32, 5, 12, 4), i(32, 6, 12, 2), i(33, 7, 1),
            i(43, 3, 11, 0), i(43, 3, 12, 1), i(43, 2, 13, 10), i(43, 2, 14, 11),
            i(43, 2, 15, 31), i(43, 2, 16, 7), i(43, 2, 17, 50), i(43, 2, 18, 99), i(59, 5, 8, 12),
        ] {
            words += instruction
        }
        if earlyBreak { words += i(20, 19) + i(43, 2, 33, .max) }
        for instruction in [
            i(54, 1, 10, 0, 7), i(248, 20), i(65, 6, 25, 8, 11), i(61, 2, 26, 25), i(247, 24, 0),
            [UInt32(labels.count + 3) << 16 | 251, 26, 22] + labels,
            i(248, 21), i(128, 2, 27, 13, 14),
        ] {
            words += instruction
        }
        if earlyBreak {
            words += i(170, 19, 34, 26, 33) + i(247, 35, 0) + i(250, 34, 36, 35)
            words += i(248, 36) + i(249, 24) + i(248, 35)
        }
        for instruction in [
            i(249, 22), i(248, 22), i(245, 2, 28, 15, 20, 27, earlyBreak ? 35 : 21), i(128, 2, 29, 28, 16), i(249, 23),
            i(248, 23), i(245, 2, 30, 17, 20, 29, 22), i(249, 24),
            i(248, 24), earlyBreak ? i(245, 2, 31, 18, 20, 30, 23, 13, 36) : i(245, 2, 31, 18, 20, 30, 23),
            i(65, 6, 32, 8, 12), i(62, 32, 31), i(253), i(56),
        ] {
            words += instruction
        }
        return words
    }

    @Test func spirvSwitchPreservesGroupedLabelsAndDefaultFallthrough() throws {
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(Self.switchModule())).module
        let function = try module.functions.value(for: module.entryPoints[0].function)
        let switches = function.statements.values.compactMap { statement -> [IRSwitchCase]? in
            if case .switchValue(_, let cases) = statement.kind { return cases }
            return nil
        }
        let selection = try #require(switches.first { $0.contains(where: \.fallThrough) })
        try #require(selection.map(\.value) == [.i32(-1), .i32(0), .default, .i32(2), .i32(9)])
        #expect(selection[0].body == selection[1].body)
        #expect(selection.map(\.fallThrough) == [true, true, true, false, false])
    }

    @Test func spirvSwitchAllowsAnUnreachableMergeWhenAllCasesReturn() throws {
        var words = Self.pre
        for instruction in [
            Self.i(15, 5, 10, 0x6e69616d, 0), Self.i(16, 10, 17, 1, 1, 1),
            Self.i(19, 1), Self.i(21, 2, 32, 0), Self.i(43, 2, 3, 1), Self.i(33, 4, 1),
            Self.i(54, 1, 10, 0, 4), Self.i(248, 11), Self.i(247, 14, 0), Self.i(251, 3, 12, 1, 13),
            Self.i(248, 12), Self.i(253), Self.i(248, 13), Self.i(253), Self.i(248, 14), Self.i(255), Self.i(56),
        ] {
            words += instruction
        }
        let compiler = IndependentShaderCompiler()
        let shader = try compiler.compileCompute(.spirv(words), to: .wgsl)
        guard case .text(.wgsl, let code) = shader.artifact else { Issue.record("Expected WGSL"); return }
        _ = try compiler.parseAndValidate(.wgsl(code))
    }

    @Test func spirvSwitchRejectsMalformedOperandsAndTargets() throws {
        try fails(Self.switchModule(labels: [0, 21, 2]))
        try fails(Self.switchModule(labels: [0, 21, 0, 21, 2, 23, 9, 24]))
        try fails(Self.switchModule(labels: [0, 20, 2, 23, 9, 24]))
        try fails(Self.switchModule(labels: [0, 21, 2, 23, UInt32.max, 21, 9, 24]))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvSwitchTranslatesEdgePhiStores(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.switchModule()), to: format)
    guard case .text(let actual, let code) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code)) }
    if format == .glsl { _ = try compiler.parseAndValidate(.glsl(code, stage: .compute, defines: [:])) }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvSwitchExecutesThePhiAssignmentFromTheSelectedEdgeOnMetal(_ earlyBreak: Bool) async throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.switchModule(earlyBreak: earlyBreak))
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let data = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
        let pipeline = try await device.createComputePipeline(shader)
        var commands: [BufferCommand] = []
        for selector: UInt32 in [.max, 0, 2, 9, 17] {
            commands += [
                .write(buffer: data, offset: 0, words: [selector]),
                .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data], workgroups: .init(x: 1)),
                .copy(source: data, sourceOffset: 4, destination: readback, byteCount: 4),
                .read(buffer: readback, byteCount: 4),
            ]
        }
        #expect(try await device.submitCommands(commands).wait() == [[earlyBreak ? 10 : 28], [28], [50], [99], [38]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
