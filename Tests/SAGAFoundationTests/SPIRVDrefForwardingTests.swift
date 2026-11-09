import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func drefForwardingModule(pointerParameters: Bool, mixedSampler: Bool = false) -> [UInt32] {
        let imageParameter: UInt32 = pointerParameters ? 9 : 6
        let samplerParameter: UInt32 = pointerParameters ? 10 : 7
        var words = pre + i(15, 5, 70, 0x6e69616d, 0) + i(16, 70, 17, 1, 1, 1)
        words += i(71, 11, 2) + i(72, 11, 0, 35, 0)
        for binding: UInt32 in 0..<7 {
            words += i(71, 30 + binding, 34, 0) + i(71, 30 + binding, 33, binding)
        }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 2), i(23, 5, 3, 4),
            i(25, 6, 3, 1, 2, 0, 0, 1, 0), i(26, 7), i(27, 8, 6), i(32, 9, 0, 6), i(32, 10, 0, 7),
            i(30, 11, 5), i(32, 12, 12, 11), i(32, 13, 12, 5), i(33, 14, 3, imageParameter, samplerParameter, 3),
            i(33, 15, 1), i(43, 2, 20, 0), i(43, 3, 21, 0), i(43, 3, 22, 0x3f000000), i(43, 3, 23, 0x3e000000),
            i(44, 4, 24, 22, 22),
            i(59, 9, 30, 0), i(59, 10, 31, 0), i(59, 9, 32, 0), i(59, 10, 33, 0), i(59, 9, 34, 0), i(59, 10, 35, 0), i(59, 12, 36, 12),
            i(54, 3, 40, 0, 14), i(55, imageParameter, 41), i(55, samplerParameter, 42), i(55, 3, 43), i(248, 44),
        ] { words += instruction }
        if pointerParameters { words += i(61, 6, 45, 41) + i(61, 7, 46, 42) }
        words += i(86, 8, 47, pointerParameters ? 45 : 41, pointerParameters ? 46 : 42)
        words += i(90, 3, 48, 47, 24, 43, 2, 21) + i(254, 48) + i(56)

        // The inner helper is called by two wrappers with distinct resource bindings.
        for function: UInt32 in [50, 60] {
            words += i(54, 3, function, 0, 14)
            words += i(55, imageParameter, function + 1) + i(55, samplerParameter, function + 2) + i(55, 3, function + 3)
            words += i(248, function + 4) + i(57, 3, function + 5, 40, function + 1, function + 2, function + 3)
            words += i(254, function + 5) + i(56)
        }

        words += i(54, 1, 70, 0, 15) + i(248, 71)
        if !pointerParameters {
            words += i(61, 6, 72, 32) + i(61, 7, 73, 33) + i(61, 6, 74, 34) + i(61, 7, 75, 35)
        }
        words += i(61, 6, 76, 30) + i(61, 7, 77, 31)
        words += i(86, 8, 78, 76, 77) + i(88, 5, 79, 78, 24, 2, 21) + i(81, 3, 80, 79, 1)
        let samplerB: UInt32 = pointerParameters ? (mixedSampler ? 31 : 35) : (mixedSampler ? 77 : 75)
        words += i(57, 3, 81, 50, pointerParameters ? 32 : 72, pointerParameters ? 33 : 73, 22)
        words += i(57, 3, 82, 60, pointerParameters ? 34 : 74, samplerB, 22)
        words += i(57, 3, 83, 60, pointerParameters ? 34 : 74, samplerB, 23)
        words += i(80, 5, 84, 81, 82, 83, 80) + i(65, 13, 85, 36, 20) + i(62, 85, 84)
        return words + i(253) + i(56)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [true, false])
func spirvDrefForwardingPreservesResourceBindings(_ format: ShaderOutputFormat, _ pointerParameters: Bool) throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.drefForwardingModule(pointerParameters: pointerParameters))
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.sampledTextures.map(\.binding.binding) == [0, 2, 4])
    #expect(shader.sampledTextures.map(\.isDepth) == [false, true, true])
    #expect(shader.samplerBindings.map(\.binding.binding) == [1, 3, 5])
    #expect(shader.samplerBindings.map(\.isComparison) == [false, true, true])
    #expect(shader.storageBindings.map(\.binding.binding) == [6])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(input)
    }
}

@Test(arguments: [true, false])
func spirvDrefForwardingRejectsSamplerWithMixedRoles(_ pointerParameters: Bool) throws {
    let words = SPIRVFrontendTests.drefForwardingModule(pointerParameters: pointerParameters, mixedSampler: true)
    do {
        _ = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
        Issue.record("Accepted a sampler shared by regular and comparison operations")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        let diagnostic = try #require(diagnostics.first)
        guard case .binary(let span) = diagnostic.location else { Issue.record("Missing SPIR-V diagnostic location"); return }
        try span.validate(wordCount: words.count)
        #expect(diagnostic.code == .unsupportedConstruct)
        #expect(diagnostic.message.contains("regular and comparison"))
    }
}

#if canImport(Metal)
@Test(arguments: [true, false])
func spirvDrefForwardingUsesEachCallersDepthAndSamplerOnMetal(_ pointerParameters: Bool) async throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.drefForwardingModule(pointerParameters: pointerParameters))
    let compiler = IndependentShaderCompiler()
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compute = try await device.createComputePipeline(compiler.compileCompute(input, to: .msl))
        let clear = ShaderInput.wgsl("""
            @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 0, 0, 1); }
            """)
        let render = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(clear, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(clear, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm, depthState: .init(compare: .always))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination]))
        let depthDescriptor = TextureDescriptor(width: 1, height: 1, format: .depth32Float, usage: [.renderAttachment, .textureBinding])
        let depthA = try await device.createTexture(depthDescriptor)
        let depthB = try await device.createTexture(depthDescriptor)
        let sampler = try await device.createSampler(.init())
        let less = try await device.createSampler(.init(compare: .less))
        let greater = try await device.createSampler(.init(compare: .greater))
        let upload = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let colorClear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        let snapshots = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff00ff00]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: color, region: .init(width: 1, height: 1)),
            .render(pipeline: render, target: target, clear: colorClear, vertexCount: 0, depth: .init(texture: depthA, loadAction: .clear(0.75))),
            .render(pipeline: render, target: target, clear: colorClear, vertexCount: 0, depth: .init(texture: depthB, loadAction: .clear(0.25))),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 6): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): color, .init(binding: 2): depthA, .init(binding: 4): depthB],
                             samplers: [.init(binding: 1): sampler, .init(binding: 3): less, .init(binding: 5): greater]),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        let expected: [Float] = [1, 1, 0, 1]
        #expect(snapshots == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
