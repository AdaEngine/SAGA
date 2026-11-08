import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private func renderBindingArraySource(unbounded: Bool = false) -> String {
    let imageCount = unbounded ? "" : ", count"
    let pairCount = unbounded ? "" : ", 2"
    return """
    enable wgpu_binding_array;
    override count = 2u;
    @group(0) @binding(0) var extra: texture_2d<f32>;
    @group(0) @binding(3) var images: binding_array<texture_2d<f32>\(imageCount)>;
    @group(0) @binding(4) var states: binding_array<sampler\(pairCount)>;
    @group(1) @binding(2) var targets: binding_array<texture_storage_2d<rgba8unorm, write>\(pairCount)>;

    struct Varyings {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(flat) red: f32,
    }

    @vertex fn vs(@builtin(vertex_index) index: u32) -> Varyings {
        let xy = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        let red = textureSampleLevel(images[0], states[0], vec2f(1.25, 0.5), 0.0).x;
        return Varyings(vec4f(xy[index], 0, 1), red + textureLoad(extra, vec2i(0), 0).x);
    }

    fn sample(image: texture_2d<f32>, state: sampler) -> f32 {
        return textureSample(image, state, vec2f(1.25, 0.5)).x;
    }

    @fragment fn fs(input: Varyings) -> @location(0) vec4f {
        let index = u32(input.position.x);
        let green = sample(images[index], states[index]);
        let color = vec4f(input.red, green, 0, 1);
        textureStore(targets[index], vec2u(0), color);
        return color;
    }
    """
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .wgsl])
func renderBindingArraysReflectSharedResources(format: ShaderOutputFormat) throws {
    let source = ShaderInput.wgsl(renderBindingArraySource())
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: format, constants: ["count": 3])
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: format, constants: ["count": 3])
    let vertexArray = try #require(vertex.sampledTextures.first { $0.binding.binding == 3 })
    let fragmentArray = try #require(fragment.sampledTextures.first)
    #expect(vertexArray.bindingArray?.count == 3)
    #expect(vertexArray.bindingArray == fragmentArray.bindingArray)
    #expect(vertexArray.samplingSamplers == [.init(group: 0, binding: 4)])
    #expect(fragmentArray.samplingSamplers == vertexArray.samplingSamplers)
    #expect(fragment.storageTextures.first?.bindingArray?.count == 2)
    #expect(vertex.storageTextures.isEmpty)
    if format == .msl {
        #expect(vertexArray.flatBindingIndex == 1)
        #expect(fragmentArray.flatBindingIndex == 0)
    } else if format == .hlsl {
        let sharedRegister = try #require(vertexArray.bindingArray?.hlslRegisterBase)
        for stage in [vertex, fragment] {
            guard case .text(.hlsl, let text) = stage.artifact else {
                Issue.record("Expected HLSL")
                continue
            }
            #expect(text.contains("[3] : register(t\(sharedRegister), space0)"))
        }
    } else {
        let capabilities = ShaderCapabilities.referenceDefault.union([
            .textureAndSamplerBindingArray, .textureAndSamplerBindingArrayNonUniformIndexing,
            .storageTextureBindingArray, .storageTextureBindingArrayNonUniformIndexing,
        ])
        for stage in [vertex, fragment] {
            guard case .text(.wgsl, let text) = stage.artifact else {
                Issue.record("Expected WGSL")
                continue
            }
            #expect(text.contains("enable wgpu_binding_array;"))
            _ = try compiler.parseAndValidate(.wgsl(text), capabilities: capabilities)
        }
    }
}

@Test func renderBindingArrayCountsSpecializePerEntryPoint() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        override count: u32;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>, count>;
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f { return textureLoad(images[1], vec2i(0), 0); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    #expect(vertex.sampledTextures.isEmpty)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    }
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl, constants: ["count": 2])
    #expect(fragment.sampledTextures.first?.bindingArray?.count == 2)
}

#if canImport(Metal)
@Test(arguments: [false, true])
func renderBindingArraysShareElementsAcrossMetalStages(unbounded: Bool) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(renderBindingArraySource(unbounded: unbounded))
    let images = IRResourceBinding(group: 0, binding: 3)
    let lengths: [IRResourceBinding: UInt32] = unbounded ? [
        images: 2, .init(group: 0, binding: 4): 2, .init(group: 1, binding: 2): 2,
    ] : [:]
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl, bindingArrayLengths: lengths)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl, bindingArrayLengths: lengths)
    let widerLengths = unbounded ? lengths.merging([images: 3], uniquingKeysWith: { _, new in new }) : [:]
    let wider = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl,
                                                constants: unbounded ? [:] : ["count": 3],
                                                bindingArrayLengths: widerLengths)
    await #expect(throws: GPUError.validation("Render stages disagree on the binding array length")) {
        try await device.createRenderPipeline(vertex: vertex, fragment: wider, colorFormat: .rgba8Unorm)
    }
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
    let input = TextureDescriptor(width: 2, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination])
    let output = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.storageBinding, .copySource])
    let a = try await device.createTexture(input)
    let b = try await device.createTexture(input)
    let first = try await device.createTexture(output)
    let second = try await device.createTexture(output)
    let target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let clamp = try await device.createSampler(.init())
    let repeating = try await device.createSampler(.init(addressModeU: .repeat))
    let readback = try await device.createBuffer(.init(size: 8, usage: [.mapRead, .copyDestination]))
    let textures: [ComputeBindingSlot: ResourceID] = [
        .init(binding: 0): a, .init(binding: 3, arrayElement: 0): a, .init(binding: 3, arrayElement: 1): b,
        .init(group: 1, binding: 2, arrayElement: 0): first, .init(group: 1, binding: 2, arrayElement: 1): second,
    ]
    let samplers: [ComputeBindingSlot: ResourceID] = [
        .init(binding: 4, arrayElement: 0): clamp, .init(binding: 4, arrayElement: 1): repeating,
    ]
    var aliased = textures
    aliased[.init(group: 1, binding: 2, arrayElement: 1)] = first
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: aliased, samplers: samplers),
        ])
    }
    let region = TextureCopyRegion(width: 2, height: 1)
    let words = try await device.submitCommands([
        .writeTexture(destination: a, data: [10, 0, 0, 255, 70, 0, 0, 255], region: region),
        .writeTexture(destination: b, data: [150, 0, 0, 255, 220, 0, 0, 255], region: region),
        .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: textures, samplers: samplers),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ]).wait()
    #expect(words == [[0xff004650, 0xff009650], [0xff004650], [0xff009650]])
    await device.close()
}
#endif
