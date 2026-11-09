import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    enum ImageGatherFailure: CaseIterable, Sendable {
        case dynamicComponent, outOfRangeComponent, colorResult, comparisonResult
    }

    static func imageGatherModule(failure: ImageGatherFailure? = nil) -> [UInt32] {
        var words = pre + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        words += i(71, 14, 2) + i(72, 14, 0, 35, 0) + i(72, 14, 1, 35, 16)
        for binding: UInt32 in 0..<5 {
            words += i(71, 30 + binding, 34, 0) + i(71, 30 + binding, 33, binding)
        }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 2), i(23, 5, 3, 4),
            i(25, 6, 3, 1, 0, 0, 0, 1, 0), i(25, 7, 3, 1, 1, 0, 0, 1, 0), i(26, 8), i(27, 9, 6), i(27, 10, 7),
            i(32, 11, 0, 6), i(32, 12, 0, 7), i(32, 13, 0, 8), i(30, 14, 5, 5), i(32, 15, 12, 14), i(32, 16, 12, 5), i(33, 17, 1),
            i(21, 18, 32, 1), i(23, 19, 18, 2), i(43, 2, 20, 0), i(43, 2, 21, 1), i(43, 2, 22, 4),
            i(43, 3, 23, 0x3f000000), i(44, 4, 24, 23, 23), i(43, 18, 25, 0), i(43, 18, 26, 1), i(44, 19, 27, 26, 25),
            i(59, 11, 30, 0), i(59, 13, 31, 0), i(59, 12, 32, 0), i(59, 13, 33, 0), i(59, 15, 34, 12),
            i(54, 1, 40, 0, 17), i(248, 41),
            i(61, 6, 43, 30), i(61, 8, 44, 31), i(61, 7, 45, 32), i(61, 8, 46, 33),
            i(86, 9, 47, 43, 44), i(86, 10, 48, 45, 46),
        ] { words += instruction }
        if failure == .dynamicComponent { words += i(128, 2, 42, 20, 21) }
        let component: UInt32 = failure == .dynamicComponent ? 42 : failure == .outOfRangeComponent ? 22 : 21
        words += i(96, failure == .colorResult ? 3 : 5, 49, 47, 24, component, 8, 27)
        words += i(97, failure == .comparisonResult ? 4 : 5, 50, 48, 24, 23)
        words += i(65, 16, 51, 34, 20) + i(62, 51, 49) + i(65, 16, 52, 34, 21) + i(62, 52, 50)
        return words + i(253) + i(56)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvImageGatherReflectsOrdinaryAndComparisonBindings(_ format: ShaderOutputFormat) throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.imageGatherModule())
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.sampledTextures.map(\.binding.binding) == [0, 2])
    #expect(shader.sampledTextures.map(\.isDepth) == [false, true])
    #expect(shader.samplerBindings.map(\.binding.binding) == [1, 3])
    #expect(shader.samplerBindings.map(\.isComparison) == [false, true])
    #expect(shader.storageBindings.map(\.binding.binding) == [4])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains("textureGather(") && source.contains("textureGatherCompare("))
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(input)
    }
}

@Test(arguments: SPIRVFrontendTests.ImageGatherFailure.allCases)
func spirvImageGatherRejectsInvalidComponentAndResult(_ failure: SPIRVFrontendTests.ImageGatherFailure) throws {
    let words = SPIRVFrontendTests.imageGatherModule(failure: failure)
    do {
        _ = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
        Issue.record("Accepted an invalid image gather")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        let diagnostic = try #require(diagnostics.first)
        guard case .binary(let span) = diagnostic.location else { Issue.record("Missing SPIR-V diagnostic location"); return }
        try span.validate(wordCount: words.count)
        try #require(words.indices.contains(span.startWord))
        #expect(words[span.startWord] & 0xffff == (failure == .comparisonResult ? 97 : 96))
    }
}

#if canImport(Metal)
@Test func spirvImageGatherPreservesColorAndComparisonLaneOrderOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let input = ShaderInput.spirv(SPIRVFrontendTests.imageGatherModule())
        let compute = try await device.createComputePipeline(compiler.compileCompute(input, to: .msl))
        let gridShader = ShaderInput.wgsl("""
            @vertex fn vs(@location(0) point: vec3f) -> @builtin(position) vec4f { return vec4f(point, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 0, 0, 1); }
            """)
        let render = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(gridShader, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(gridShader, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm,
            vertexBuffers: [.init(stride: 12, attributes: [.init(location: 0, offset: 0, format: .float32x3)])],
            depthState: .init())
        let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
        let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba16Float, usage: [.textureBinding, .copyDestination]))
        let depth = try await device.createTexture(.init(width: 2, height: 2, format: .depth32Float, usage: [.renderAttachment, .textureBinding]))
        let sampler = try await device.createSampler(.init(addressModeU: .repeat))
        let compare = try await device.createSampler(.init(compare: .less))
        let upload = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .copySource]))
        let vertices = try await device.createBuffer(.init(size: 288, usage: [.vertex, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let x: [Float] = [-1, 0, 1]
        let y: [Float] = [1, 0, -1]
        let values: [[Float]] = [[0.125, 0.75], [0.625, 0.25]]
        var grid: [Float] = []
        for row in 0..<2 {
            for column in 0..<2 {
                let (left, right, top, bottom, z) = (x[column], x[column + 1], y[row], y[row + 1], values[row][column])
                grid += [left, top, z, left, bottom, z, right, top, z, right, top, z, left, bottom, z, right, bottom, z]
            }
        }
        let snapshots = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0x30003c00, 0, 0x34003c00, 0]),
            .write(buffer: upload, offset: 256, words: [0x38003c00, 0, 0x3a003c00, 0]),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: color, region: .init(width: 2, height: 2)),
            .write(buffer: vertices, offset: 0, words: grid.map(\.bitPattern)),
            .render(pipeline: render, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1),
                    vertexCount: 24, vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth)),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 4): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): color, .init(binding: 2): depth],
                             samplers: [.init(binding: 1): sampler, .init(binding: 3): compare]),
            .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
        ]).wait()
        let expected: [Float] = [0.75, 0.5, 0.125, 0.25, 1, 0, 1, 0]
        #expect(snapshots == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
