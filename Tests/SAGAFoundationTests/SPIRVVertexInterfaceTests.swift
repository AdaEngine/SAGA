import Testing
import SAGACompiler
import SAGAFrontends
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func vertexInterfaceModule(blockOutput: Bool = true, writePointSize: Bool = false) -> [UInt32] {
        var words = pre + i(15, 0, 10, 0x6e69616d, 0, 11, 12, 13)
        words += i(71, 11, 11, 42) + i(71, 13, 30, 1) + i(71, 13, 14)
        if blockOutput {
            words += i(71, 8, 2) + i(72, 8, 0, 11, 0) + i(72, 8, 1, 11, 1)
            words += i(72, 8, 2, 11, 3) + i(72, 8, 3, 11, 4)
        } else {
            words += i(71, 12, 11, 0)
        }
        for instruction in [
            i(19, 1), i(20, 2), i(21, 3, 32, 1), i(22, 4, 32), i(23, 5, 4, 4),
            i(32, 6, 1, 3), i(32, 7, 3, 5), i(32, 9, 3, 4), i(33, 14, 1),
            i(43, 3, 20, 0), i(43, 3, 21, 1), i(43, 3, 22, 2),
            i(43, 4, 23, Float(-1).bitPattern), i(43, 4, 24, Float(1).bitPattern),
            i(43, 4, 25, 0), i(43, 4, 26, Float(90).bitPattern),
        ] { words += instruction }
        if blockOutput {
            words += i(28, 15, 4, 21) + i(30, 8, 5, 4, 15, 15) + i(32, 16, 3, 8)
        }
        for instruction in [
            i(59, 6, 11, 1), i(59, blockOutput ? 16 : 7, 12, 3), i(59, 9, 13, 3),
            i(54, 1, 10, 0, 14), i(248, 30), i(61, 3, 31, 11),
            i(170, 2, 32, 31, 21), i(170, 2, 33, 31, 22),
            i(169, 4, 34, 32, 24, 23), i(169, 4, 35, 33, 24, 23),
            i(80, 5, 36, 34, 35, 25, 24),
        ] { words += instruction }
        if blockOutput { words += i(65, 7, 37, 12, 20) }
        if writePointSize { words += i(65, 9, 38, 12, 21) + i(62, 38, 24) }
        return words + i(62, blockOutput ? 37 : 12, 36) + i(62, 13, 26) + i(253) + i(56)
    }

    @Test func spirvVertexInterfaceRejectsWritesToUnsupportedBlockBuiltins() throws {
        try fails(Self.vertexInterfaceModule(writePointSize: true), code: .unsupportedConstruct)
    }

    @Test func spirvVertexInterfaceAdjustsYOnlyWhenRequested() throws {
        for adjustment in [false, true] {
            let module = try parse(Self.vertexInterfaceModule(), options: .init(adjustCoordinateSpace: adjustment))
            let entry = try #require(module.entryPoints.first)
            let function = try module.functions.value(for: entry.function)
            let flipsY = function.expressions.values.contains {
                if case .unary(.negate, _) = $0.kind { return true }
                return false
            }
            #expect(flipsY == adjustment)
        }
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func spirvVertexInterfaceTranslatesDirectAndBlockOutputs(_ format: ShaderOutputFormat, _ block: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(
        .spirv(SPIRVFrontendTests.vertexInterfaceModule(blockOutput: block)), entryPoint: "main", to: format)
    guard case .text(let actual, let text) = shader.artifact else { Issue.record("Expected shader text"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

#if canImport(Metal)
@Test func spirvVertexInterfaceRendersPositionAndVaryingOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.spirv(SPIRVFrontendTests.vertexInterfaceModule()), entryPoint: "main", to: .msl)
    let fragment = try compiler.compileRenderStage(.spirv(SPIRVFrontendTests.fragmentLocationsModule()), entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                           usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: pipeline, target: texture, clear: .init(red: 0, green: 0, blue: 0, alpha: 0), vertexCount: 3),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 264),
        ]).wait()
        let words = try #require(result.first)
        try #require(words.count == 66)
        #expect(words[0] == 0xff004039)
        #expect(words[65] == 0)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
