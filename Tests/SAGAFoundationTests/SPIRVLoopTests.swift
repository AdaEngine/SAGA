import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func structuredLoopModule(postTest: Bool = false) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 10, 0x6e69616d, 0), i(16, 10, 17, 1, 1, 1),
            i(71, 5, 2), i(72, 5, 0, 35, 0), i(71, 8, 34, 0), i(71, 8, 33, 0),
            i(19, 1), i(20, 2), i(21, 3, 32, 0), i(30, 5, 3), i(32, 6, 12, 5), i(32, 7, 12, 3), i(33, 9, 1),
            i(43, 3, 11, 0), i(43, 3, 12, 1), i(43, 3, 13, 6), i(43, 3, 14, 10), i(59, 6, 8, 12),
            i(54, 1, 10, 0, 9), i(248, 20), i(249, 21),
            i(248, 21), i(245, 3, 32, 11, 20, 41, 28), i(245, 3, 33, 11, 20, 40, 28),
            i(246, 29, 28, 0), i(249, 22), i(248, 22),
        ] {
            words += instruction
        }
        if postTest { words += i(249, 23) }
        else { words += i(176, 2, 34, 32, 13) + i(250, 34, 23, 29) }
        for instruction in [
            i(248, 23), i(247, 27, 0), i(251, 32, 26, 1, 24, postTest ? 9 : 3, 25),
            i(248, 24), i(249, 28), i(248, 25), i(249, 29), i(248, 26), i(249, 27),
            i(248, 27), i(128, 3, 37, 32, 14), i(128, 3, 38, 33, 37), i(249, 28),
            i(248, 28), i(245, 3, 40, 33, 24, 38, 27), i(128, 3, 41, 32, 12),
        ] {
            words += instruction
        }
        if postTest {
            words += i(176, 2, 42, 41, 13) + i(250, 42, 21, 29)
            words += i(248, 29) + i(245, 3, 43, 33, 25, 40, 28)
        } else {
            words += i(249, 21) + i(248, 29) + i(245, 3, 43, 33, 22, 33, 25)
        }
        return words + i(65, 7, 44, 8, 11) + i(62, 44, 43) + i(253) + i(56)
    }

    @Test func spirvStructuredLoopKeepsSwitchAndLoopJumpTargetsSeparate() throws {
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(Self.structuredLoopModule())).module
        let function = try module.functions.value(for: module.entryPoints[0].function)
        #expect(function.statements.values.contains { if case .switchValue = $0.kind { true } else { false } })
        #expect(function.statements.values.contains { if case .continue = $0.kind { true } else { false } })
        #expect(function.statements.values.contains { if case .break = $0.kind { true } else { false } })
    }

    @Test func spirvStructuredLoopAcceptsHeaderAsSingleBlockContinueTarget() throws {
        var words = Self.pre
        for instruction in [
            Self.i(20, 1), Self.i(21, 2, 32, 0), Self.i(43, 2, 3, 0), Self.i(43, 2, 4, 1),
            Self.i(43, 2, 5, 3), Self.i(33, 6, 2), Self.i(54, 2, 10, 0, 6),
            Self.i(248, 11), Self.i(249, 12), Self.i(248, 12),
            Self.i(245, 2, 13, 3, 11, 14, 12), Self.i(128, 2, 14, 13, 4), Self.i(176, 1, 15, 14, 5),
            Self.i(246, 16, 12, 0), Self.i(250, 15, 12, 16), Self.i(248, 16), Self.i(254, 13), Self.i(56),
        ] {
            words += instruction
        }
        let function = try IndependentShaderCompiler().parseAndValidate(.spirv(words)).module.functions.values[0]
        #expect(function.statements.values.contains { if case .loop = $0.kind { true } else { false } })
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvStructuredLoopTranslatesSeparateConditionsAndBackedges(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for postTest in [false, true] {
        let result = try compiler.compileCompute(.spirv(SPIRVFrontendTests.structuredLoopModule(postTest: postTest)), to: format)
        guard case .text(let actual, let text) = result.artifact else { Issue.record("Expected shader text"); continue }
        #expect(actual == format)
        if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
        if format == .glsl { _ = try compiler.parseAndValidate(.glsl(text, stage: .compute, defines: [:])) }
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvStructuredLoopRunsSwitchContinueAndLoopBreakOnMetal(_ postTest: Bool) async throws {
    let shader = try IndependentShaderCompiler().compileCompute(
        .spirv(SPIRVFrontendTests.structuredLoopModule(postTest: postTest)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 4),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[postTest ? 64 : 22]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
