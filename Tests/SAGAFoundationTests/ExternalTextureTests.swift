import Testing
import SAGACompiler
import SAGAShaderIR

private let externalTextureCompute = """
    @group(0) @binding(0) var first: texture_external;
    @group(1) @binding(0) var second: texture_external;
    @group(0) @binding(1) var state: sampler;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4f, 4>;

    fn sample(image: texture_external, s: sampler, x: f32) -> vec4f {
        return textureSampleBaseClampToEdge(image, s, vec2f(x, 0.5));
    }
    fn forward(image: texture_external, s: sampler, x: f32) -> vec4f {
        return sample(image, s, x);
    }
    @compute @workgroup_size(1) fn run() {
        output[0] = forward(first, state, -0.25);
        output[1] = forward(second, state, 1.25);
        output[2] = textureLoad(first, vec2u(1, 0));
        output[3] = vec4f(vec2f(textureDimensions(first)), vec2f(textureDimensions(second)));
    }
    """

private let externalTextureRender = """
    @group(0) @binding(0) var frame: texture_external;
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        let xy = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return vec4f(xy[i], 0, 1);
    }
    @fragment fn fs(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
        return textureLoad(frame, vec2i(i32(pixel.x), 0));
    }
    """

@Test func externalTextureWGSLReflectsOpaqueBindingsAcrossHelpers() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(externalTextureCompute)
    let capabilities = ShaderCapabilities.referenceDefault.union(.textureExternal)
    let module = try compiler.parseAndValidate(source, capabilities: capabilities).module
    let expected: IRTypeKind = .image(dimension: .d2, arrayed: false, class: .external)
    for global in module.globalVariables.values.prefix(2) {
        #expect(global.space == .handle)
        #expect(try module.types.value(for: global.type).kind == expected)
    }
    let shader = try compiler.compileWebGPUCompute(source)
    #expect(shader.externalTextures == [.init(group: 0, binding: 0), .init(group: 1, binding: 0)])
    #expect(shader.sampledTextures.isEmpty)
    #expect(shader.samplerBindings.count == 1)
    guard case .text(.wgsl, let text) = shader.artifact else {
        Issue.record("Expected WGSL")
        return
    }
    #expect(text.split(separator: "\n").filter { $0.contains(": texture_external;") }.count == 2)
    #expect(text.split(separator: "\n").filter { $0.contains("textureSampleBaseClampToEdge(") }.count == 2)
    _ = try compiler.parseAndValidate(.wgsl(text), capabilities: capabilities)
}

@Test func externalTextureRenderReflectionTracksOnlyReachableStages() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(externalTextureRender)
    let vertex = try compiler.compileWebGPURenderStage(source, entryPoint: "vs")
    let fragment = try compiler.compileWebGPURenderStage(source, entryPoint: "fs")
    #expect(vertex.externalTextures.isEmpty)
    #expect(fragment.externalTextures == [.init(group: 0, binding: 0)])
    guard case .text(.wgsl, let text) = fragment.artifact else {
        Issue.record("Expected WGSL")
        return
    }
    _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.textureExternal))
}

@Test func externalTextureRequiresCapability() throws {
    let input = ShaderInput.wgsl(externalTextureCompute)
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Expected a capability diagnostic")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .unsupportedCapability)
    }
}

@Test(arguments: [
    "textureSample(frame, state, vec2f(0.5))",
    "textureSampleLevel(frame, state, vec2f(0.5), 0.0)",
    "textureLoad(frame, vec2i(0), 0)",
    "textureDimensions(frame, 0)",
    "textureNumLevels(frame)",
    "textureSampleBaseClampToEdge(frame, comparison, vec2f(0.5))",
])
func externalTextureRejectsInvalidWGSLBuiltins(_ expression: String) throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var frame: texture_external;
        @group(0) @binding(1) var state: sampler;
        @group(0) @binding(2) var comparison: sampler_comparison;
        @compute @workgroup_size(1) fn run() { _ = \(expression); }
        """)
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source, capabilities: .referenceDefault.union(.textureExternal))
        Issue.record("Expected a type diagnostic")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .typeMismatch)
    }
}

@Test func externalTexturesCannotBeBindingArrayElements() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var frames: binding_array<texture_external, 2>;
        @compute @workgroup_size(1) fn run() {}
        """)
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source, capabilities: .allKnown)
        Issue.record("Expected an invalid type diagnostic")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .invalidType)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func externalTextureLoweringReflectsNativePlanesAndParameters(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(externalTextureCompute), to: format)
    #expect(shader.externalTextureLayouts.map(\.binding) == shader.externalTextures)
    #expect(shader.externalTextureLayouts.count == 2)
    #expect(ComputeShaderArtifact.ExternalTextureLayout.parameterByteSize == 208)
    #expect(shader.externalTextureLayouts.allSatisfy { $0.planeBindingIndices.count == 3 })
    if format != .hlsl {
        #expect(Set(shader.externalTextureLayouts.flatMap(\.planeBindingIndices)).count == 6)
        #expect(Set(shader.externalTextureLayouts.map(\.parametersBindingIndex)).count == 2)
    }
    if format == .glsl {
        #expect(shader.glslTextureSamplerBindings.count == 6)
        #expect(shader.glslTextureSamplerBindings.allSatisfy { $0.externalTexturePlane != nil })
        #expect(Set(shader.glslTextureSamplerBindings.map(\.flatBindingIndex)).count == 6)
    }
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected text")
        return
    }
    #expect(source.contains("saga_external_convert"))
    #expect(source.contains("saga_external_sample"))
    #expect(source.contains("saga_external_load"))
    let render = ShaderInput.wgsl(externalTextureRender)
    let vertex = try compiler.compileRenderStage(render, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(render, entryPoint: "fs", to: format)
    #expect(vertex.externalTextureLayouts.isEmpty)
    #expect(fragment.externalTextureLayouts.first?.planeBindingIndices.count == 3)
}

#if canImport(Metal)
import Metal

@Test func externalTextureMSLConvertsPlanesAndCropsOnMetal() throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        @group(0) @binding(0) var rgba: texture_external;
        @group(0) @binding(1) var interleaved: texture_external;
        @group(0) @binding(2) var separate: texture_external;
        @group(0) @binding(3) var state: sampler;
        @group(0) @binding(4) var<storage, read_write> output: array<vec4f, 6>;
        fn sample(frame: texture_external, s: sampler, x: f32) -> vec4f {
            return textureSampleBaseClampToEdge(frame, s, vec2f(x, 0.5));
        }
        @compute @workgroup_size(1) fn run() {
            output[0] = sample(rgba, state, -0.5);
            output[1] = textureLoad(rgba, vec2i(0));
            output[2] = vec4f(vec2f(textureDimensions(rgba)), 0.0, 0.0);
            output[3] = sample(interleaved, state, 0.25);
            output[4] = textureLoad(separate, vec2i(1, 0));
            output[5] = textureLoad(separate, vec2i(-1, 0));
        }
        """), to: .msl)
    guard case .text(.msl, let source) = shader.artifact else {
        Issue.record("Expected MSL")
        return
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try device.makeLibrary(source: source, options: nil)
    let pipeline = try device.makeComputePipelineState(function: #require(library.makeFunction(name: shader.entryPoint)))
    let queue = try #require(device.makeCommandQueue())
    let commands = try #require(queue.makeCommandBuffer())
    let encoder = try #require(commands.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)

    func texture<T>(_ format: MTLPixelFormat, width: Int, pixels: [T]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        pixels.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, width, 1), mipmapLevel: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: $0.count)
        }
        return texture
    }
    func parameters(planes: UInt32, cropped: Bool) -> [UInt32] {
        var words = Array(repeating: UInt32(0), count: 52)
        for index in [0, 5, 10, 15, 21, 26] { words[index] = Float(1).bitPattern }
        words[16] = Float(2).bitPattern
        words.replaceSubrange(28..<32, with: [Float(1), 0, 2, 1].map(\.bitPattern))
        words.replaceSubrange(32..<36, with: [Float(1), 0, 2, 1].map(\.bitPattern))
        words.replaceSubrange(36..<42, with: [Float(cropped ? 0.5 : 1), 0, 0, 1, cropped ? 0.5 : 0, 0].map(\.bitPattern))
        words.replaceSubrange(42..<48, with: [Float(1), 0, 0, 1, cropped ? 1 : 0, 0].map(\.bitPattern))
        if cropped { words[48] = 1; words[49] = 1 }
        words[50] = planes
        return words
    }

    let rgba = try texture(.rgba8Unorm, width: 2, pixels: [UInt8(255), 0, 0, 255, 0, 255, 0, 255])
    let y = try texture(.r32Float, width: 2, pixels: [Float(0.25), 0.5])
    let uv = try texture(.rg32Float, width: 1, pixels: [Float(0.5), 0.75])
    let cb = try texture(.r32Float, width: 1, pixels: [Float(0.5)])
    let cr = try texture(.r32Float, width: 1, pixels: [Float(0.75)])
    let planes = [[rgba, cb, cr], [y, uv, cr], [y, cb, cr]]
    for layout in shader.externalTextureLayouts {
        let index = Int(layout.binding.binding)
        for (texture, slot) in zip(planes[index], layout.planeBindingIndices) {
            encoder.setTexture(texture, index: Int(slot))
        }
        let words = parameters(planes: UInt32(index + 1), cropped: index == 0)
        words.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: Int(layout.parametersBindingIndex))
        }
    }
    let samplerDescriptor = MTLSamplerDescriptor()
    samplerDescriptor.sAddressMode = .repeat
    let sampler = try #require(device.makeSamplerState(descriptor: samplerDescriptor))
    let samplerSlot = try #require(shader.samplerBindings.first?.flatBindingIndex)
    encoder.setSamplerState(sampler, index: Int(samplerSlot))
    let output = try #require(device.makeBuffer(length: 96, options: .storageModeShared))
    let outputSlot = try #require(shader.storageBindings.first?.flatBindingIndex)
    encoder.setBuffer(output, offset: 0, index: Int(outputSlot))
    let single = MTLSize(width: 1, height: 1, depth: 1)
    encoder.dispatchThreadgroups(single, threadsPerThreadgroup: single)
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    #expect(commands.status == .completed)
    let actual = UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: 24)
    let expected: [Float] = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0,
                             0.3535534, 0.5, 0.75, 1, 0.7071068, 0.5, 0.75, 1, 0, 0, 0, 0]
    #expect(zip(actual, expected).allSatisfy { abs($0.0 - $0.1) < 0.001 })
}
#endif
