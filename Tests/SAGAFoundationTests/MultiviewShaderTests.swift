import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal
#endif

private let multiviewShader = ShaderInput.wgsl("""
    struct VertexInput {
        @builtin(vertex_index) index: u32,
        @builtin(view_index) view: u32,
    }
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(flat) view: u32,
    }
    @vertex fn vs(input: VertexInput) -> VertexOutput {
        let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return VertexOutput(vec4f(points[input.index], 0, 1), input.view);
    }
    @fragment fn fs(@builtin(view_index) view: u32, @location(0) @interpolate(flat) vertexView: u32)
        -> @location(0) vec4u {
        return vec4u(vertexView + 10u, view + 20u, 0u, 255u);
    }
    """)

private func multiviewText(_ stage: RenderStageArtifact) throws -> String {
    let source: String? = if case .text(_, let source) = stage.artifact { source } else { nil }
    return try #require(source)
}

private func multiviewBinary(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(multiviewShader, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func multiviewShaderExportsSPIRVStageInputsAndCapability() throws {
    let words = try multiviewBinary(IndependentShaderCompiler())
    try #require(words.count > 5)
    var capabilities: Set<UInt32> = []
    var extensions: Set<String> = []
    var views: Set<UInt32> = []
    var flat: Set<UInt32> = []
    var interfaces: [UInt32: Set<UInt32>] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 10:
            let bytes = words[(offset + 1)..<(offset + count)].flatMap { word in
                (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) }
            }
            extensions.insert(String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self))
        case 15 where count >= 4: // Both fixture entry names fit in one word.
            interfaces[words[offset + 1]] = Set(words[(offset + 4)..<(offset + count)])
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 71 where count == 4 && words[offset + 2] == 11 && words[offset + 3] == 4440:
            views.insert(words[offset + 1])
        case 71 where count == 3 && words[offset + 2] == 14: flat.insert(words[offset + 1])
        default: break
        }
        offset += count
    }
    #expect(capabilities.contains(4439)) // MultiView
    #expect(words[1] >= 0x00010300 || extensions.contains("SPV_KHR_multiview"))
    #expect(views.count == 2)
    let vertex = try #require(interfaces[0]).intersection(views)
    let fragment = try #require(interfaces[4]).intersection(views)
    #expect(vertex.count == 1 && fragment.count == 1 && vertex.isDisjoint(with: fragment))
    #expect(fragment.isSubset(of: flat))
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl], [false, true])
func multiviewShaderEmitsViewIndexForBothStages(_ format: ShaderOutputFormat, _ roundTrip: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = try roundTrip ? ShaderInput.spirv(multiviewBinary(compiler)) : multiviewShader
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: format)
    for stage in [vertex, fragment] {
        #expect(stage.readsViewIndex)
        let source = try multiviewText(stage)
        switch format {
        case .msl: #expect(source.contains("[[amplification_id]]"))
        case .hlsl:
            #expect(source.contains("SV_ViewID"))
            #expect(stage.hlslProfile == (stage.stage == .vertex ? "vs_6_8" : "ps_6_1"))
        case .glsl:
            #expect(source.contains("#extension GL_EXT_multiview : require"))
            #expect(source.contains("uint(gl_ViewIndex)"))
        case .wgsl:
            #expect(source.contains("@builtin(view_index)"))
            _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.multiview])
        default: Issue.record("Unexpected output target")
        }
    }
    #expect(fragment.fragmentInputBuiltinSlots == 0)
}

@Test func multiviewShaderChecksCapabilityTypeStageAndWebGPUProfile() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(multiviewShader, capabilities: []) }
    for source in [
        "@fragment fn f(@builtin(view_index) view: i32) {}",
        "@compute @workgroup_size(1) fn f(@builtin(view_index) view: u32) {}",
        "@fragment fn f() -> @builtin(view_index) u32 { return 0u; }",
        "@fragment fn f(@builtin(view_index) a: u32, @builtin(view_index) b: u32) {}",
    ] {
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(source), capabilities: [.multiview]) }
    }
    for entry in ["vs", "fs"] {
        #expect(throws: ShaderFailure.self) { try compiler.compileWebGPURenderStage(multiviewShader, entryPoint: entry) }
    }
    let vertex = try compiler.compileRenderStage(.wgsl("""
        @vertex fn f(@builtin(view_index) view: u32) -> @builtin(position) vec4f {
            return vec4f(f32(view), 0, 0, 1);
        }
        """), entryPoint: "f", to: .hlsl)
    #expect(vertex.hlslProfile == "vs_6_1")
}

@Test func multiviewShaderRequiresRuntimeSupportAtPipelineCreation() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(multiviewShader, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(multiviewShader, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: RenderTestBackend())
    await #expect(throws: GPUError.validation("Feature not advertised")) {
        try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
    }
    await #expect(throws: GPUError.validation("Feature not advertised")) {
        try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, multiviewMask: 3)
    }
    await device.close()
}

#if canImport(Metal)
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsVertexAmplificationCount(4) == true),
      arguments: [UInt32(0b1010), 0b0100, 0b1111], [false, true])
func multiviewQueueRoutesViewsAndKeepsAttachmentOperationsLayered(_ mask: UInt32, _ roundTrip: Bool) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.multiview, .selectiveMultiview])
    do {
        let compiler = IndependentShaderCompiler()
        let input = try roundTrip ? ShaderInput.spirv(multiviewBinary(compiler)) : multiviewShader
        let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .msl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment,
                                                             colorFormat: .rgba8Uint, multiviewMask: mask)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
            usage: [.renderAttachment, .copySource, .copyDestination], arrayLayerCount: 6))
        let view = try await device.createTextureView(texture: texture, .init(
            dimension: .d2Array, baseArrayLayer: 1, arrayLayerCount: 4))
        let short = try await device.createTextureView(texture: texture, .init(
            dimension: .d2Array, baseArrayLayer: 1, arrayLayerCount: 1))
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.render(pipeline: pipeline, target: short, colorLoad: .load, vertexCount: 3)])
        }
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let arguments = try await device.createBuffer(.init(size: 16, usage: [.indirect, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let clear = RenderColorLoadAction.clear(.init(red: 5, green: 6, blue: 7, alpha: 8))
        var commands: [BufferCommand] = (0..<6).map { layer in
            .writeTexture(destination: texture, data: [1, 2, 3, 4], region: .init(width: 1, height: 1, arrayLayer: layer))
        }
        switch mask {
        case 0b0100:
            commands += [
                .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
                .renderIndexed(pipeline: pipeline, target: view, colorLoad: clear,
                               indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3)),
            ]
        case 0b1111:
            commands += [
                .write(buffer: arguments, offset: 0, words: [3, 1, 0, 0]),
                .renderIndirect(pipeline: pipeline, target: view, colorLoad: clear, indirectBuffer: arguments),
            ]
        default:
            commands.append(.render(pipeline: pipeline, target: view, colorLoad: clear, vertexCount: 3))
        }
        let copies = (0..<6).flatMap { (layer: UInt32) -> [BufferCommand] in
            [.copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, arrayLayer: layer),
             .read(buffer: read, byteCount: 4)]
        }
        commands += copies
        commands.append(.render(pipeline: pipeline, target: view, colorLoad: .load, vertexCount: 0, colorStore: .discard))
        commands += copies
        let snapshots = try await device.submitCommands(commands).wait()
        let rendered = (0..<4).map { layer -> UInt32 in
            mask & (1 << layer) == 0 ? 0x08070605 : 0xff00140a + UInt32(layer) * 0x101
        }
        #expect(snapshots == ([0x04030201] + rendered + [0x04030201, 0x04030201, 0, 0, 0, 0, 0x04030201]).map { [$0] })
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsVertexAmplificationCount(4) == true))
func multiviewQueueWritesDepthInSelectedLayersOnly() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.multiview, .selectiveMultiview])
    do {
        let vertex = try IndependentShaderCompiler().compileRenderStage(multiviewShader, entryPoint: "vs", to: .msl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: nil, colorFormat: nil,
            depthState: .init(format: .depth32Float, compare: .always, writeEnabled: true), multiviewMask: 0b0101)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: [.renderAttachment, .copySource], arrayLayerCount: 4))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = [.render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 3,
            depth: .init(texture: texture, loadAction: .clear(0.75)))]
        for layer: UInt32 in 0..<4 {
            commands += [
                .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256,
                                     aspect: .depthOnly, arrayLayer: layer),
                .read(buffer: read, byteCount: 4),
            ]
        }
        #expect(try await device.submitCommands(commands).wait() == [[0], [Float(0.75).bitPattern], [0], [Float(0.75).bitPattern]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsVertexAmplificationCount(2) == true), arguments: [false, true])
func multiviewShaderReadsAmplifiedViewIDsOnMetal(_ roundTrip: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = try roundTrip ? ShaderInput.spirv(multiviewBinary(compiler)) : multiviewShader
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .msl)
    let device = try #require(MTLCreateSystemDefaultDevice())
    try #require(device.supportsVertexAmplificationCount(2))
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = try device.makeLibrary(source: multiviewText(vertex), options: nil)
        .makeFunction(name: vertex.entryPoint)
    descriptor.fragmentFunction = try device.makeLibrary(source: multiviewText(fragment), options: nil)
        .makeFunction(name: fragment.entryPoint)
    descriptor.maxVertexAmplificationCount = 2
    descriptor.inputPrimitiveTopology = .triangle
    descriptor.colorAttachments[0].pixelFormat = .rgba8Uint
    let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
    let texture = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Uint, width: 1, height: 1, mipmapped: false)
    texture.textureType = .type2DArray
    texture.arrayLength = 2
    texture.usage = .renderTarget
    texture.storageMode = .shared
    let target = try #require(device.makeTexture(descriptor: texture))
    let pass = MTLRenderPassDescriptor()
    pass.renderTargetArrayLength = 2
    pass.colorAttachments[0].texture = target
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].storeAction = .store
    let command = try #require(device.makeCommandQueue()?.makeCommandBuffer())
    let encoder = try #require(command.makeRenderCommandEncoder(descriptor: pass))
    encoder.setRenderPipelineState(pipeline)
    let views: [MTLVertexAmplificationViewMapping] = [
        .init(viewportArrayIndexOffset: 0, renderTargetArrayIndexOffset: 0),
        .init(viewportArrayIndexOffset: 0, renderTargetArrayIndexOffset: 1),
    ]
    views.withUnsafeBufferPointer { encoder.setVertexAmplificationCount(2, viewMappings: $0.baseAddress) }
    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try #require(command.status == .completed, "\(String(describing: command.error))")
    var words = [UInt32](repeating: 0, count: 2)
    for slice in words.indices {
        target.getBytes(&words[slice], bytesPerRow: 4, bytesPerImage: 4, from: MTLRegionMake2D(0, 0, 1, 1),
                        mipmapLevel: 0, slice: slice)
    }
    #expect(words == [0xff00140a, 0xff00150b])
}
#endif
