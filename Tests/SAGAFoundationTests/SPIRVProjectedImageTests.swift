import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    enum ProjectedImageFailure: CaseIterable, Sendable {
        case missingDivisor, arrayed, cube
    }

    static func projectedImageModule(fragment: Bool, failure: ProjectedImageFailure? = nil) -> [UInt32] {
        var words = pre
        if fragment {
            words += i(15, 4, 40, 0x6e69616d, 0, 34) + i(16, 40, 7) + i(71, 34, 30, 0)
        } else {
            words += i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
            words += i(71, 14, 2) + i(72, 14, 0, 35, 0) + i(71, 34, 34, 0) + i(71, 34, 33, 4)
        }
        for binding: UInt32 in 0..<4 {
            words += i(71, 30 + binding, 34, 0) + i(71, 30 + binding, 33, binding)
        }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 2), i(23, 5, 3, 4),
            i(25, 6, 3, failure == .cube ? 3 : 1, 0, failure == .arrayed ? 1 : 0, 0, 1, 0),
            i(25, 7, 3, 1, 1, 0, 0, 1, 0), i(27, 8, 6), i(27, 9, 7), i(26, 10),
            i(32, 11, 0, 6), i(32, 12, 0, 7), i(32, 13, 0, 10),
        ] { words += instruction }
        words += fragment ? i(32, 15, 3, 5) : i(30, 14, 5) + i(32, 15, 12, 14) + i(32, 16, 12, 5)
        for instruction in [
            i(33, 17, 1), i(43, 2, 20, 0), i(43, 3, 21, 0), i(43, 3, 22, 0x3f400000),
            i(43, 3, 23, 0x3f000000), i(43, 3, 24, 0x40000000), i(43, 3, 25, 0x3e800000),
            i(43, 3, 26, 0x3f800000), i(44, 5, 27, 22, 23, 24, 25), i(44, 4, 28, 22, 23),
            i(59, 11, 30, 0), i(59, 13, 31, 0), i(59, 12, 32, 0), i(59, 13, 33, 0),
            i(59, 15, 34, fragment ? 3 : 12),
            i(54, 1, 40, 0, 17), i(248, 41),
            i(61, 6, 42, 30), i(61, 10, 43, 31), i(61, 7, 44, 32), i(61, 10, 45, 33),
            i(86, 8, 46, 42, 43), i(86, 9, 47, 44, 45),
        ] { words += instruction }

        // Dividing u=0.75 by q=2 moves sampling from the green texel to the red texel.
        // The unused fourth lane is 0.25, so taking the last lane as q also samples green.
        let coordinate: UInt32 = failure == .missingDivisor ? 28 : 27
        words += fragment ? i(91, 5, 48, 46, coordinate) : i(92, 5, 48, 46, coordinate, 2, 21)
        // The projected reference 1/2 passes the less comparison with 0.75.
        words += fragment ? i(93, 3, 49, 47, 27, 26) : i(94, 3, 49, 47, 27, 26, 2, 21)
        words += i(81, 3, 50, 48, 0) + i(81, 3, 51, 48, 1) + i(80, 5, 52, 50, 51, 49, 26)
        words += fragment ? i(62, 34, 52) : i(65, 16, 53, 34, 20) + i(62, 53, 52)
        return words + i(253) + i(56)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvProjectedImagesTranslate(_ format: ShaderOutputFormat, _ fragment: Bool) throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.projectedImageModule(fragment: fragment))
    let compiler = IndependentShaderCompiler()
    let artifact: ShaderArtifact
    if fragment {
        let shader = try compiler.compileRenderStage(input, entryPoint: "main", to: format)
        #expect(shader.sampledTextures.map(\.isDepth) == [false, true])
        #expect(shader.samplerBindings.map(\.isComparison) == [false, true])
        artifact = shader.artifact
    } else {
        let shader = try compiler.compileCompute(input, to: format)
        #expect(shader.sampledTextures.map(\.isDepth) == [false, true])
        #expect(shader.samplerBindings.map(\.isComparison) == [false, true])
        artifact = shader.artifact
    }
    if format == .wgsl {
        guard case .text(.wgsl, let source) = artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: SPIRVFrontendTests.ProjectedImageFailure.allCases)
func spirvProjectedImagesRejectInvalidShapes(_ failure: SPIRVFrontendTests.ProjectedImageFailure) throws {
    let words = SPIRVFrontendTests.projectedImageModule(fragment: false, failure: failure)
    do {
        _ = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
        Issue.record("Accepted an invalid projected image sample")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        let diagnostic = try #require(diagnostics.first)
        guard case .binary(let span) = diagnostic.location else { Issue.record("Missing SPIR-V diagnostic location"); return }
        try span.validate(wordCount: words.count)
        try #require(words.indices.contains(span.startWord))
        #expect(words[span.startWord] & 0xffff == 92)
        #expect(diagnostic.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func spirvProjectedImagesDividePositionAndDepthReferenceOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let computeInput = ShaderInput.spirv(SPIRVFrontendTests.projectedImageModule(fragment: false))
    let fragmentInput = ShaderInput.spirv(SPIRVFrontendTests.projectedImageModule(fragment: true))
    let baseRender = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            let positions = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return vec4f(positions[index], 0, 1);
        }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 0, 0, 1); }
        """)
    let vertex = try compiler.compileRenderStage(baseRender, entryPoint: "vs", to: .msl)
    let clear = try compiler.compileRenderStage(baseRender, entryPoint: "fs", to: .msl)
    let projected = try compiler.compileRenderStage(fragmentInput, entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compute = try await device.createComputePipeline(compiler.compileCompute(computeInput, to: .msl))
        let render = try await device.createRenderPipeline(vertex: vertex, fragment: projected, colorFormat: .rgba8Unorm)
        let clearDepth = try await device.createRenderPipeline(vertex: vertex, fragment: clear, colorFormat: .rgba8Unorm,
                                                               depthState: .init(compare: .always))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.textureBinding, .copyDestination]))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding]))
        let sampler = try await device.createSampler()
        let compare = try await device.createSampler(.init(compare: .less))
        let upload = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let textures: [ComputeBindingSlot: ResourceID] = [.init(binding: 0): color, .init(binding: 2): depth]
        let samplers: [ComputeBindingSlot: ResourceID] = [.init(binding: 1): sampler, .init(binding: 3): compare]
        let snapshots = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ff00]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: color, region: .init(width: 2, height: 1)),
            .render(pipeline: clearDepth, target: target, clear: .transparentBlack, vertexCount: 0,
                    depth: .init(texture: depth, loadAction: .clear(0.75))),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 4): output], workgroups: .init(x: 1),
                             textures: textures, samplers: samplers),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: textures, samplers: samplers),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(snapshots == [[Float(1).bitPattern, 0, Float(1).bitPattern, Float(1).bitPattern], [0xffff00ff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
