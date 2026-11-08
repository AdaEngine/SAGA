import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func forwardLoopModule(metadata: Bool = true, recursive: Bool = false) -> [UInt32] {
        var words = head + i(17, 1)
        if metadata { words += i(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0) }
        words += i(14, 0, 1) + i(15, 5, 10, 0x6e69616d, 0) + i(16, 10, 17, 1, 1, 1)
        if metadata {
            words += i(7, 81, 0x6c736c67, 0) + i(3, 2, 450, 81, 0x20)
            words += i(5, 20, 0x6574616c, 0x72) + i(6, 4, 0, 0x64726f77, 0)
        }
        for instruction in [
            i(71, 4, 2), i(72, 4, 0, 35, 0), i(71, 8, 34, 0), i(71, 8, 33, 0),
            i(19, 1), i(20, 2), i(21, 3, 32, 0), i(30, 4, 3), i(32, 5, 12, 4), i(32, 6, 12, 3), i(32, 7, 7, 3),
            i(33, 9, 1), i(33, 11, 1, 7), i(43, 3, 12, 0), i(43, 3, 13, 1), i(43, 3, 14, 3), i(59, 5, 8, 12),
            i(54, 1, 10, 0, 9), i(248, 21), i(59, 7, 22, 7), i(62, 22, 12), i(57, 1, 23, 20, 22),
            i(61, 3, 24, 22), i(65, 6, 25, 8, 12), i(62, 25, 24), i(253), i(56),
            i(54, 1, 20, 0, 11), i(55, 7, 26), i(248, 27), i(249, 28),
            i(248, 28), i(246, 31, 30, 0), i(249, 29),
            i(248, 29), i(61, 3, 32, 26), i(128, 3, 33, 32, 13), i(62, 26, 33), i(249, 30),
            i(248, 30), i(61, 3, 34, 26), i(176, 2, 35, 34, 14), i(250, 35, 28, 31), i(248, 31),
        ] {
            words += instruction
        }
        if recursive { words += i(57, 1, 36, 10) }
        return words + i(253) + i(56)
    }

    @Test func spirvFunctionOrderKeepsEntryAndMemberNamesAfterReordering() throws {
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(Self.forwardLoopModule())).module
        #expect(module.entryPoints.first?.function.index == 1)
        #expect(module.functions.values.first?.name == "later")
        let output = try #require(module.globalVariables.values.first)
        guard case .structure(let fields, _) = try module.types.value(for: output.type).kind else {
            Issue.record("Expected storage structure")
            return
        }
        #expect(fields.first?.name == "word")
    }

    @Test func spirvFunctionOrderRejectsRecursiveForwardCalls() throws {
        try fails(Self.forwardLoopModule(metadata: false, recursive: true))
    }

    @Test func spirvFunctionOrderRejectsSourceFileIdsThatAreNotStrings() throws {
        try fails(Self.pre + Self.i(3, 2, 450, 1) + Self.i(19, 1))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func spirvFunctionOrderTranslatesForwardCalls(_ format: ShaderOutputFormat, _ metadata: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.spirv(SPIRVFrontendTests.forwardLoopModule(metadata: metadata))
    let shader = try compiler.compileCompute(source, to: format)
    guard case .text(let actual, let text) = shader.artifact else { Issue.record("Expected shader text"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

#if canImport(Metal)
@Test func spirvFunctionOrderRunsForwardCalledDoWhileOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.forwardLoopModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 4),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[3]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
