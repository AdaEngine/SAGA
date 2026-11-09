import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal
#endif

@Test func multisampledArrayRequiresAnAdvertisedFeature() throws {
    let descriptor = TextureDescriptor(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment,
                                       sampleCount: 4, arrayLayerCount: 2)
    #expect(throws: GPUError.validation("Feature not advertised")) {
        try descriptor.validate(against: .init(maxBufferSize: 1024))
    }
    try descriptor.validate(against: .init(maxBufferSize: 1024, features: [.multisampleArray]))
    let layers = TextureViewDescriptor(dimension: .d2Array, baseArrayLayer: 1, arrayLayerCount: 1)
    #expect(try layers.arrayLayers(in: descriptor) == 1..<2)
    #expect(throws: GPUError.self) {
        try TextureViewDescriptor(dimension: .cube).arrayLayers(in: descriptor)
    }
}

#if canImport(Metal)
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsVertexAmplificationCount(4) == true),
      arguments: [UInt32(0b101), 0b010])
func multisampledArrayResolvesAllAttachmentLayersOnMetal(_ mask: UInt32) async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return vec4f(points[index], 0, 1);
        }
        @fragment fn fs(@builtin(view_index) view: u32, @builtin(sample_index) sample: u32) -> @location(0) vec4f {
            return vec4f(f32(view + 1u) * 0.25, f32(sample) * 0.25, 0, 1);
        }
        @group(0) @binding(0) var image: texture_multisampled_2d<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 4>;
        @compute @workgroup_size(4) fn readSamples(@builtin(local_invocation_index) sample: u32) {
            let texel = textureLoad(image, vec2i(0), sample);
            output[sample] = u32(round(texel.g * 255.0)) | (u32(round(texel.b * 255.0)) << 8u);
        }
        """)
    let device = try await Device.make(backend: MetalBackend.makeDevice(),
                                      requiredFeatures: [.multiview, .selectiveMultiview, .multisampleArray])
    do {
        let pipeline = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm, sampleCount: 4, multiviewMask: mask)
        let sampleReader = try await device.createComputePipeline(compiler.compileCompute(source, entryPoint: "readSamples", to: .msl))
        let descriptor = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .textureBinding, .copySource, .copyDestination], sampleCount: 4, arrayLayerCount: 5)
        let samples = try await device.createTexture(descriptor)
        let copiedSamples = try await device.createTexture(descriptor)
        let resolved = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource, .copyDestination], arrayLayerCount: 5))
        let range = TextureViewDescriptor(dimension: .d2Array, baseArrayLayer: 1, arrayLayerCount: 3)
        let target = try await device.createTextureView(texture: samples, range)
        let destination = try await device.createTextureView(texture: resolved, range)
        let copiedLayer = try await device.createTextureView(texture: copiedSamples, .init(dimension: .d2, baseArrayLayer: 2))
        let copiedClearLayer = try await device.createTextureView(texture: copiedSamples, .init(dimension: .d2, baseArrayLayer: 3))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let readLayers = (0..<5).flatMap { (layer: UInt32) -> [BufferCommand] in
            [.copyTextureToBuffer(source: resolved, destination: readback, bytesPerRow: 256, arrayLayer: layer),
             .read(buffer: readback, byteCount: 4)]
        }
        let firstRenderedLayer = UInt32(mask.trailingZeroBitCount) + 1
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        await #expect(throws: GPUError.validation("Texture copy subresources must be disjoint")) {
            try await device.submitCommands([.copyTextureToTexture(source: samples, destination: samples,
                sourceRegion: .init(width: 1, height: 1, depth: 2), destinationOrigin: .init(arrayLayer: 1))])
        }
        var commands: [BufferCommand] = [
            .writeTexture(destination: resolved, data: [1, 2, 3, 4], region: .init(width: 1, height: 1)),
            .writeTexture(destination: resolved, data: [5, 6, 7, 8], region: .init(width: 1, height: 1, arrayLayer: 4)),
            .render(pipeline: pipeline, target: target, clear: blue, vertexCount: 3, resolve: .init(texture: destination)),
            .copyTextureToTexture(source: samples, destination: copiedSamples,
                                  sourceRegion: .init(width: 1, height: 1, arrayLayer: firstRenderedLayer, depth: 2),
                                  destinationOrigin: .init(arrayLayer: 2)),
            .dispatchCompute(pipeline: sampleReader, bindings: [.init(binding: 1): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): copiedLayer]),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .dispatchCompute(pipeline: sampleReader, bindings: [.init(binding: 1): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): copiedClearLayer]),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]
        commands += readLayers
        commands.append(.render(pipeline: pipeline, target: target, colorLoad: .load, vertexCount: 0,
                                resolve: .init(texture: destination), colorStore: .discard))
        commands += readLayers
        commands.append(.render(pipeline: pipeline, target: target, colorLoad: .load, vertexCount: 0,
                                resolve: .init(texture: destination)))
        commands += readLayers
        let snapshots = try await device.submitCommands(commands).wait()
        let rendered = [UInt32(0xff006040), 0xff006080, 0xff0060bf].enumerated().map { layer, color in
            mask & (1 << layer) != 0 ? color : 0xffff0000
        }
        let colors = [0x04030201] + rendered + [0x08070605]
        #expect(snapshots == [[0, 64, 128, 191], [0xff00, 0xff00, 0xff00, 0xff00]] +
                (colors + colors + [0x04030201, 0, 0, 0, 0x08070605]).map { [$0] })
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsVertexAmplificationCount(4) == true))
func multisampledArrayPreservesDepthSamplesByViewOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return vec4f(points[index], 0.25, 1);
        }
        @group(0) @binding(0) var selected: texture_depth_multisampled_2d;
        @group(0) @binding(1) var cleared: texture_depth_multisampled_2d;
        @group(0) @binding(2) var<storage, read_write> output: array<vec2f, 4>;
        @compute @workgroup_size(4) fn readSamples(@builtin(local_invocation_index) sample: u32) {
            output[sample] = vec2f(textureLoad(selected, vec2i(0), sample), textureLoad(cleared, vec2i(0), sample));
        }
        """)
    let device = try await Device.make(backend: MetalBackend.makeDevice(),
                                      requiredFeatures: [.multiview, .selectiveMultiview, .multisampleArray])
    do {
        let pipeline = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl), fragment: nil, colorFormat: nil,
            depthState: .init(compare: .always), sampleCount: 4, sampleMask: 0b1010, multiviewMask: 0b101)
        let sampleReader = try await device.createComputePipeline(compiler.compileCompute(source, entryPoint: "readSamples", to: .msl))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding], sampleCount: 4, arrayLayerCount: 3))
        let selected = try await device.createTextureView(texture: depth, .init(aspect: .depthOnly, dimension: .d2, baseArrayLayer: 2))
        let cleared = try await device.createTextureView(texture: depth, .init(aspect: .depthOnly, dimension: .d2, baseArrayLayer: 1))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .clear(0.75))),
            .dispatchCompute(pipeline: sampleReader, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): selected, .init(binding: 1): cleared]),
            .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
        ]).wait()
        let expected = [Float(0.75), 0.75, 0.25, 0.75, 0.75, 0.75, 0.25, 0.75]
        #expect(snapshots == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
