import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func clipDistanceModule(count: UInt32, block: Bool = true, capability: Bool = true) -> [UInt32] {
        var words = head + i(17, 1)
        if capability { words += i(17, 32) }
        words += i(14, 0, 1)
        if block {
            words += i(15, 0, 14, 0x6e69616d, 0, 15, 16, 18)
            words += i(71, 9, 2) + i(72, 9, 0, 11, 0) + i(72, 9, 1, 11, 3)
        } else {
            words += i(15, 0, 14, 0x6e69616d, 0, 15, 16, 17, 18)
            words += i(71, 16, 11, 0) + i(71, 17, 11, 3)
        }
        words += i(71, 15, 30, 0) + i(71, 18, 30, 0)
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 2), i(23, 5, 3, 4),
            i(43, 2, 20, 0), i(43, 2, 21, 1), i(43, 2, 22, count), i(43, 2, 23, count - 1),
            i(43, 3, 24, 0), i(43, 3, 25, Float(1).bitPattern),
            i(28, 8, 3, 22), i(32, 6, 1, 4), i(32, 7, 3, 5), i(32, 12, 3, 3), i(33, 13, 1),
        ] { words += instruction }
        if block { words += i(30, 9, 5, 8) + i(32, 11, 3, 9) }
        else { words += i(32, 10, 3, 8) }
        words += i(59, 6, 15, 1) + i(59, block ? 11 : 7, 16, 3) + i(59, 12, 18, 3)
        if !block { words += i(59, 10, 17, 3) }
        words += i(54, 1, 14, 0, 13) + i(248, 30)
        words += i(61, 4, 31, 15) + i(81, 3, 32, 31, 0) + i(81, 3, 33, 31, 1)
        words += i(80, 5, 34, 32, 33, 24, 25)
        if block { words += i(65, 7, 35, 16, 20) }
        words += i(62, block ? 35 : 16, 34) + i(62, 18, 25)
        func planePointer(_ result: UInt32, index: UInt32) -> [UInt32] {
            block ? i(65, 12, result, 16, 21, index) : i(65, 12, result, 17, index)
        }
        words += planePointer(37, index: 20) + i(62, 37, 32)
        if count > 1 { words += i(127, 3, 38, 33) + planePointer(39, index: 23) + i(62, 39, 38) }
        return words + i(253) + i(56)
    }

    @Test func spirvClipDistanceRejectsMissingCapabilityAndTooManyPlanes() throws {
        try fails(Self.clipDistanceModule(count: 1, block: false, capability: false), code: .unsupportedCapability)
        try fails(Self.clipDistanceModule(count: 9), code: .unsupportedConstruct)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func spirvClipDistanceTranslatesDirectAndBlockOutputs(_ format: ShaderOutputFormat, _ block: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.spirv(SPIRVFrontendTests.clipDistanceModule(count: 5, block: block))
    let shader = try compiler.compileRenderStage(source, entryPoint: "main", to: format)
    #expect(shader.clipDistanceCount == 5)
    #expect(shader.varyings.map(\.location) == [0])
    guard case .text(let actual, let code) = shader.artifact else { Issue.record("Expected text artifact"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code), capabilities: [.clipDistances]) }
    if format == .hlsl { #expect(code.contains("SV_ClipDistance1")) }
}

#if canImport(Metal)
@Test(arguments: [UInt32(1), 5, 8])
func spirvClipDistanceClipsTheExpectedPixelsOnMetal(_ count: UInt32) async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.spirv(SPIRVFrontendTests.clipDistanceModule(count: count)),
                                                 entryPoint: "main", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("""
        @fragment fn fs(@location(0) tint: f32) -> @location(0) vec4f { return vec4f(0, tint, 0, 1); }
        """), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.clipDistances])
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                           usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [Float(-1), 1, 3, 1, -1, -3].map(\.bitPattern)),
            .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 1, alpha: 1),
                    vertexCount: 3, vertexBuffers: [.init(buffer: vertices)]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 264),
        ]).wait()
        let words = try #require(result.first)
        try #require(words.count == 66)
        let blue: UInt32 = 0xffff0000
        let green: UInt32 = 0xff00ff00
        #expect([words[0], words[1], words[64], words[65]] == [blue, green, blue, count == 1 ? green : blue])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
