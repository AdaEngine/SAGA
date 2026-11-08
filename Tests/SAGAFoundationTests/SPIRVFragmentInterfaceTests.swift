import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func fragmentLocationsModule(
        omitInput: Bool = false, secondOutput: UInt32? = nil, writeInput: Bool = false
    ) -> [UInt32] {
        var words = head + i(17, 1) + i(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0) + i(14, 0, 1)
        var interfaces: [UInt32] = [12]
        if !omitInput { interfaces.append(11) }
        if secondOutput != nil { interfaces.append(13) }
        words += [UInt32(interfaces.count + 5) << 16 | 15, 4, 10, 0x6e69616d, 0] + interfaces
        words += i(16, 10, 7) + i(71, 11, 30, 1) + i(71, 11, 14) + i(71, 12, 30, 0)
        if let secondOutput { words += i(71, 13, 30, secondOutput) }
        for instruction in [
            i(19, 1), i(22, 2, 32), i(23, 3, 2, 4), i(32, 4, 1, 2), i(32, 5, 3, 3), i(33, 6, 1), i(33, 7, 2),
            i(43, 2, 20, Float(400).bitPattern), i(43, 2, 21, Float(0.25).bitPattern),
            i(43, 2, 22, 0), i(43, 2, 23, Float(1).bitPattern),
            i(59, 4, 11, 1), i(59, 5, 12, 3),
        ] { words += instruction }
        if secondOutput != nil { words += i(59, 5, 13, 3) }
        words += i(54, 1, 10, 0, 6) + i(248, 30)
        words += i(57, 2, 33, 14)
        words += i(136, 2, 34, 33, 20) + i(80, 3, 35, 34, 21, 22, 23) + i(62, 12, 35)
        if secondOutput != nil { words += i(62, 13, 35) }
        words += i(253) + i(56) + i(54, 2, 14, 0, 7) + i(248, 39)
        if writeInput { words += i(62, 11, 22) }
        return words + i(61, 2, 31, 11) + i(12, 2, 32, 80, 11, 31) + i(12, 2, 36, 80, 12, 32) + i(254, 36) + i(56)
    }

    @Test func spirvFragmentInterfaceRejectsMissingDuplicateAndWritableInputs() throws {
        try fails(Self.fragmentLocationsModule(omitInput: true))
        try fails(Self.fragmentLocationsModule(secondOutput: 0))
        try fails(Self.fragmentLocationsModule(writeInput: true))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func spirvFragmentInterfaceTranslatesLocationInputsAndOutputs(_ format: ShaderOutputFormat, _ multiple: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.fragmentLocationsModule(secondOutput: multiple ? 2 : nil))
    let shader = try compiler.compileRenderStage(input, entryPoint: "main", to: format)
    guard case .text(let actual, let text) = shader.artifact else { Issue.record("Expected shader text"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

private let locationVertexShader = """
    struct Raster {
        @builtin(position) position: vec4f,
        @location(1) @interpolate(flat) angle: f32,
    }
    @vertex fn vs(@builtin(vertex_index) index: u32) -> Raster {
        var points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return Raster(vec4f(points[index], 0, 1), 90);
    }
    """

#if canImport(Metal)
@Test func spirvFragmentInterfaceRendersTheAngleAtLocationOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(locationVertexShader), entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(.spirv(SPIRVFrontendTests.fragmentLocationsModule()), entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: pipeline, target: texture, clear: .init(red: 0, green: 0, blue: 0, alpha: 0), vertexCount: 3),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff004039]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
