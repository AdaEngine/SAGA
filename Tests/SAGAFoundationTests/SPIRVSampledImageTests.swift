import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    enum SampledImageFailure: CaseIterable, Sendable {
        case notASampler, sampleCoordinate, sampleLod, fetchCoordinate, fetchLod, queryLod, queryLevelsType, missingImageQuery

        var offendingOpcode: UInt32 {
            switch self {
            case .notASampler: 86
            case .sampleCoordinate, .sampleLod: 88
            case .fetchCoordinate, .fetchLod: 95
            case .queryLod, .missingImageQuery: 103
            case .queryLevelsType: 106
            }
        }
    }

    static func sampledImageModule(gradients: Bool = false, failure: SampledImageFailure? = nil) -> [UInt32] {
        var words = head + i(17, 1)
        if failure != .missingImageQuery { words += i(17, 50) }
        words += i(14, 0, 1) + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        words += i(71, 15, 2)
        for binding: UInt32 in 0..<3 {
            words += i(72, 15, binding, 35, binding * 16)
            words += i(71, 30 + binding, 34, 0) + i(71, 30 + binding, 33, binding)
        }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(22, 4, 32), i(23, 5, 3, 2),
            i(23, 6, 4, 2), i(23, 7, 4, 4), i(23, 8, 2, 2), i(23, 9, 2, 4),
            i(25, 10, 4, 1, 0, 0, 0, 1, 0), i(26, 11), i(27, 12, 10),
            i(32, 13, 0, 10), i(32, 14, 0, 11), i(30, 15, 9, 9, 9), i(32, 16, 12, 15),
            i(32, 17, 12, 9), i(33, 18, 1),
            i(43, 3, 20, 0), i(43, 3, 21, 1), i(43, 4, 22, 0x3f400000), i(43, 4, 23, 0x3f000000),
            i(43, 4, 24, 0x3f800000), i(44, 5, 25, 20, 20), i(44, 6, 26, 22, 23),
            i(43, 2, 27, 0), i(43, 2, 28, 1), i(43, 2, 29, 2),
        ] { words += instruction }
        if gradients {
            // These derivatives select mip 1 for the 4-by-2 texture used on Metal.
            words += i(43, 4, 33, 0) + i(44, 6, 34, 23, 33) + i(44, 6, 35, 33, 23)
        }
        for instruction in [
            i(59, 13, 30, 0), i(59, 14, 31, 0), i(59, 16, 32, 12), i(54, 1, 40, 0, 18), i(248, 41),
            i(61, 10, 42, 30), i(61, 11, 43, 31), i(86, 12, 44, 42, failure == .notASampler ? 42 : 43),
        ] { words += instruction }
        if gradients {
            words += i(88, 7, 45, 44, 26, 4, 34, 35) + i(100, 10, 57, 44)
        } else {
            words += i(88, 7, 45, 44, failure == .sampleCoordinate ? 25 : 26, 2, failure == .sampleLod ? 21 : 24)
        }
        for instruction in [
            i(95, 7, 46, gradients ? 57 : 42, failure == .fetchCoordinate ? 26 : 25, 2, failure == .fetchLod ? 24 : 21),
            i(103, 5, 47, 42, failure == .queryLod ? 24 : 21), i(106, failure == .queryLevelsType ? 5 : 3, 48, 42),
            i(124, 9, 49, 45), i(124, 9, 50, 46), i(124, 8, 51, 47), i(124, 2, 52, 48), i(80, 9, 53, 51, 52, 27),
            i(65, 17, 54, 32, 27), i(62, 54, 49), i(65, 17, 55, 32, 28), i(62, 55, 50),
            i(65, 17, 56, 32, 29), i(62, 56, 53), i(253), i(56),
        ] { words += instruction }
        return words
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvSampledImagesReflectSeparateTextureAndSampler(_ format: ShaderOutputFormat, _ gradients: Bool) throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.sampledImageModule(gradients: gradients))
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(input, to: format)
    let sampler = IRResourceBinding(group: 0, binding: 1)
    let texture = try #require(shader.sampledTextures.first)
    #expect(shader.sampledTextures.count == 1)
    #expect(texture.binding == .init(group: 0, binding: 0))
    #expect(texture.dimension == .d2 && texture.sampleKind == .float && !texture.isMultisampled)
    #expect(texture.samplingSamplers == [sampler])
    #expect(shader.samplerBindings.map(\.binding) == [sampler])
    #expect(shader.storageBindings.map(\.binding) == [.init(group: 0, binding: 2)])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(input)
    }
}

@Test(arguments: SPIRVFrontendTests.SampledImageFailure.allCases)
func spirvSampledImagesRejectInvalidOperands(_ failure: SPIRVFrontendTests.SampledImageFailure) throws {
    let words = SPIRVFrontendTests.sampledImageModule(failure: failure)
    do {
        _ = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
        Issue.record("Accepted an invalid sampled image operation")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        let diagnostic = try #require(diagnostics.first)
        guard case .binary(let span) = diagnostic.location else { Issue.record("Missing SPIR-V diagnostic location"); return }
        try span.validate(wordCount: words.count)
        try #require(words.indices.contains(span.startWord))
        #expect(words[span.startWord] & 0xffff == failure.offendingOpcode)
        if failure == .missingImageQuery { #expect(diagnostic.code == .unsupportedCapability) }
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvSampledImagesSampleFetchAndQuerySelectedMipOnMetal(_ gradients: Bool) async throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.sampledImageModule(gradients: gradients))
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let texture = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: [.textureBinding, .copyDestination], mipLevelCount: 2))
        let sampler = try await device.createSampler(.init(mipmapFilter: .nearest))
        let upload = try await device.createBuffer(.init(size: 280, usage: [.copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
        let red = Array(repeating: UInt32(0xff0000ff), count: 4)
        let snapshots = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: red),
            .write(buffer: upload, offset: 256, words: red + [0xffff0000, 0xff00ff00]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: texture,
                                 region: .init(width: 4, height: 2)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 272), destination: texture,
                                 region: .init(width: 2, height: 1, mipLevel: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): texture], samplers: [.init(binding: 1): sampler]),
            .copy(source: output, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
        ]).wait()
        let colors: [Float] = [0, 1, 0, 1, 0, 0, 1, 1]
        #expect(snapshots == [colors.map(\.bitPattern) + [2, 1, 2, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
