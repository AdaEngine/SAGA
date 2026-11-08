import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func fragmentSubgroupModule(
        laneBuiltin: UInt32 = 41, listLane: Bool = true, subgroupCapability: Bool = true, writeLane: Bool = false
    ) -> [UInt32] {
        var words = head
        for instruction in [
            i(17, 1), subgroupCapability ? i(17, 61) : [], subgroupCapability ? i(17, 64) : [], i(14, 0, 1),
            listLane ? i(15, 4, 20, 0x6e69616d, 0, 16, 17, 18) : i(15, 4, 20, 0x6e69616d, 0, 16, 18),
            i(16, 20, 7), i(71, 16, 11, 36), i(71, 16, 14), i(71, 17, 11, laneBuiltin), i(71, 17, 14),
            i(71, 18, 30, 0),
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(23, 4, 2, 4),
            i(32, 5, 1, 2), i(32, 6, 1, 3), i(32, 7, 3, 4), i(33, 8, 1), i(33, 9, 4),
            i(43, 2, 10, 3), i(43, 2, 11, 173), i(43, 3, 12, 0),
            i(59, 5, 16, 1), i(59, 6, 17, 1), i(59, 7, 18, 3),
            i(54, 1, 20, 0, 8), i(248, 21), i(57, 4, 22, 30), i(62, 18, 22), i(253), i(56),
            i(54, 4, 30, 0, 9), i(248, 31), i(57, 4, 32, 40), i(254, 32), i(56),
            i(54, 4, 40, 0, 9), i(248, 41), writeLane ? i(62, 17, 12) : [],
            i(61, 2, 42, 16), i(61, 3, 43, 17), i(124, 2, 44, 43),
            subgroupCapability ? i(338, 2, 45, 10, 44) : i(83, 2, 45, 44),
            i(80, 4, 46, 42, 44, 45, 11), i(254, 46), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvFragmentSubgroupInputsAreReadOnlyAndStageChecked() throws {
        try fails(Self.fragmentSubgroupModule(laneBuiltin: 40))
        try fails(Self.fragmentSubgroupModule(laneBuiltin: 38))
        try fails(Self.fragmentSubgroupModule(laneBuiltin: 36))
        try fails(Self.fragmentSubgroupModule(listLane: false))
        try fails(Self.fragmentSubgroupModule(subgroupCapability: false), code: .unsupportedCapability)
        try fails(Self.fragmentSubgroupModule(writeLane: true))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvFragmentSubgroupInputsTranslateThroughHelpers(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.spirv(SPIRVFrontendTests.fragmentSubgroupModule())
    let module = try compiler.parseAndValidate(source, capabilities: [.subgroup]).module
    let entry = try module.functions.value(for: module.entryPoints[0].function)
    #expect(entry.arguments.map(\.binding) == [.builtIn(.subgroupSize), .builtIn(.subgroupInvocationId)])
    #expect(module.functions.values.allSatisfy { $0.arguments.count == 2 })
    let fragment = try compiler.compileRenderStage(source, entryPoint: "main", to: format)
    #expect(fragment.requiresSubgroups)
    guard case .text(let actual, let text) = fragment.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.subgroup]) }
}

#if canImport(Metal)
@Test func spirvFragmentSubgroupInputsReachRasterizedPixelsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl("""
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return vec4f(points[i], 0, 1);
        }
        """), entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(.spirv(SPIRVFrontendTests.fragmentSubgroupModule()),
                                                   entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.subgroups])
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
        let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: pipeline, target: texture, clear: .transparentBlack, vertexCount: 3),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 512),
        ]).wait()
        let pixels = try #require(result.first)
        #expect(pixels.count == 128)
        for index in [0, 1, 64, 65] {
            let pixel = pixels[index]
            let size = pixel & 255
            let lane = pixel >> 8 & 255
            let first = pixel >> 16 & 255
            #expect(pixel >> 24 == 173)
            #expect((4...128).contains(size) && size.nonzeroBitCount == 1 && first <= lane && lane < size)
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
