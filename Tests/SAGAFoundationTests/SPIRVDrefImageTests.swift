import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    enum DrefImageFailure: CaseIterable, Sendable {
        case mixedSampler, nonDepthImage, integerReference, integerLod, nonzeroLod
    }

    enum DrefImageEncoding: CaseIterable, Sendable {
        case declaredDepth, unknownDepth, depthInPair
    }

    static func drefImageModule(encoding: DrefImageEncoding = .declaredDepth, failure: DrefImageFailure? = nil) -> [UInt32] {
        var words = pre + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        words += i(71, 14, 2) + i(72, 14, 0, 35, 0)
        for binding: UInt32 in 0..<5 {
            words += i(71, 30 + binding, 34, 0) + i(71, 30 + binding, 33, binding)
        }
        let dref: UInt32 = failure == .integerReference ? 20 : 22
        let lod: UInt32 = failure == .integerLod ? 20 : failure == .nonzeroLod ? 22 : 21
        let depthType: UInt32 = encoding == .declaredDepth ? 7 : 6
        let depthPair: UInt32 = encoding == .unknownDepth || failure == .nonDepthImage ? 8 : 9
        let depthPointer: UInt32 = encoding == .declaredDepth ? 12 : 11
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 2), i(23, 5, 3, 4),
            i(25, 6, 3, 1, encoding == .unknownDepth ? 2 : 0, 0, 0, 1, 0),
            i(27, 8, 6), i(26, 10), i(32, 11, 0, 6), i(32, 13, 0, 10),
        ] { words += instruction }
        if encoding != .unknownDepth { words += i(25, 7, 3, 1, 1, 0, 0, 1, 0) + i(27, 9, 7) }
        if encoding == .declaredDepth { words += i(32, 12, 0, 7) }
        for instruction in [
            i(30, 14, 5), i(32, 15, 12, 14), i(32, 16, 12, 5), i(33, 17, 1),
            i(43, 2, 20, 0), i(43, 3, 21, 0), i(43, 3, 22, 0x3f000000), i(43, 3, 23, 0x3f600000), i(44, 4, 24, 22, 22),
            i(59, 11, 30, 0), i(59, 13, 31, 0), i(59, depthPointer, 32, 0), i(59, 13, 33, 0), i(59, 15, 34, 12),
            i(54, 1, 40, 0, 17), i(248, 41),
            i(61, 6, 42, 30), i(61, 10, 43, 31), i(61, depthType, 44, 32), i(61, 10, 45, 33),
            i(86, 8, 46, 42, 43), i(88, 5, 48, 46, 24, 2, 21),
        ] { words += instruction }
        let comparisonImage: UInt32 = failure == .nonDepthImage ? 42 : 44
        let comparisonSampler: UInt32 = failure == .mixedSampler ? 43 : 45
        let copies = encoding == .unknownDepth
        if copies {
            // Both images share the same raw type. Follow the copies to their distinct bindings.
            words += i(83, depthType, 55, comparisonImage) + i(83, 10, 56, comparisonSampler)
        }
        for instruction in [
            i(86, depthPair, 47, copies ? 55 : comparisonImage, copies ? 56 : comparisonSampler),
            i(90, 3, 49, 47, 24, dref, 2, lod), i(90, 3, 50, 47, 24, 23, 2, 21),
            i(81, 3, 51, 48, 0), i(81, 3, 52, 48, 1), i(80, 5, 53, 49, 50, 51, 52),
            i(65, 16, 54, 34, 20), i(62, 54, 53), i(253), i(56),
        ] { words += instruction }
        return words
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], SPIRVFrontendTests.DrefImageEncoding.allCases)
func spirvDrefImagesKeepRegularAndComparisonSamplersSeparate(
    _ format: ShaderOutputFormat, _ encoding: SPIRVFrontendTests.DrefImageEncoding
) throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.drefImageModule(encoding: encoding))
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.sampledTextures.map(\.binding) == [IRResourceBinding(group: 0, binding: 0), .init(group: 0, binding: 2)])
    #expect(shader.sampledTextures.map(\.isDepth) == [false, true])
    #expect(shader.samplerBindings.map(\.binding) == [IRResourceBinding(group: 0, binding: 1), .init(group: 0, binding: 3)])
    #expect(shader.samplerBindings.map(\.isComparison) == [false, true])
    #expect(shader.storageBindings.map(\.binding) == [.init(group: 0, binding: 4)])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains("textureSampleCompareLevel("))
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(input)
    }
}

@Test(arguments: SPIRVFrontendTests.DrefImageFailure.allCases)
func spirvDrefImagesRejectInvalidComparisonOperands(_ failure: SPIRVFrontendTests.DrefImageFailure) throws {
    let words = SPIRVFrontendTests.drefImageModule(failure: failure)
    do {
        _ = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
        Issue.record("Accepted an invalid depth comparison")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        let diagnostic = try #require(diagnostics.first)
        guard case .binary(let span) = diagnostic.location else { Issue.record("Missing SPIR-V diagnostic location"); return }
        try span.validate(wordCount: words.count)
        try #require(words.indices.contains(span.startWord))
        if failure == .mixedSampler {
            #expect(diagnostic.message.lowercased().contains("sampler"))
        } else {
            #expect(words[span.startWord] & 0xffff == 90)
        }
        if failure == .integerReference { #expect(diagnostic.code == .typeMismatch) }
        if failure == .nonzeroLod { #expect(diagnostic.code == .unsupportedConstruct) }
    }
}

#if canImport(Metal)
@Test(arguments: SPIRVFrontendTests.DrefImageEncoding.allCases)
func spirvDrefImagesCompareDepthAndSampleColorOnMetal(_ encoding: SPIRVFrontendTests.DrefImageEncoding) async throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.drefImageModule(encoding: encoding))
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
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.textureBinding, .copyDestination]))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding]))
        let sampler = try await device.createSampler(.init())
        let compare = try await device.createSampler(.init(compare: .less))
        let upload = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff00ff00]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: color, region: .init(width: 1, height: 1)),
            .render(pipeline: render, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 0,
                    depth: .init(texture: depth, loadAction: .clear(0.75))),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 4): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): color, .init(binding: 2): depth],
                             samplers: [.init(binding: 1): sampler, .init(binding: 3): compare]),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        let expected: [Float] = [1, 0, 0, 1]
        #expect(snapshots == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
