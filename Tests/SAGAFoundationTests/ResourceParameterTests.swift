import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let forwardedResourceShader = """
    requires texture_and_sampler_let;
    @group(0) @binding(0) var firstImage: texture_2d<f32>;
    @group(1) @binding(0) var secondImage: texture_2d<f32>;
    @group(0) @binding(1) var clampState: sampler;
    @group(1) @binding(1) var repeatState: sampler;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @group(2) @binding(0) var firstTarget: texture_storage_2d<rgba8unorm, write>;
    @group(2) @binding(1) var secondTarget: texture_storage_2d<rgba8unorm, write>;

    fn read(image: texture_2d<f32>, state: sampler, x: f32) -> f32 {
        let width = textureDimensions(image).x;
        return textureSampleLevel(image, state, vec2f(x, 0.5), 0.0).x * f32(width) * 0.5;
    }
    fn forward(x: f32, state: sampler, image: texture_2d<f32>) -> f32 {
        return read(image, state, x);
    }
    fn save(target: texture_storage_2d<rgba8unorm, write>, red: f32) {
        textureStore(target, vec2u(0), vec4f(red, 0, 0, 1));
    }
    @compute @workgroup_size(1) fn run() {
        let a = forward(1.25, clampState, firstImage);
        let imageAlias = firstImage;
        let samplerAlias = repeatState;
        let b = forward(1.25, samplerAlias, imageAlias);
        let c = forward(1.25, clampState, secondImage);
        let d = forward(1.25, repeatState, secondImage);
        out = vec4f(a, b, c, d);
        save(firstTarget, a);
        save(secondTarget, c);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func resourceParametersSpecializeEachGlobalTextureSamplerPair(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(.wgsl(forwardedResourceShader))
    let shader = try compiler.compileCompute(.wgsl(forwardedResourceShader), to: format)
    #expect(shader.sampledTextures.count == 2)
    #expect(shader.sampledTextures.allSatisfy { $0.samplingSamplers.count == 2 })
    #expect(shader.samplerBindings.count == 2)
    #expect(shader.storageTextures.count == 2)
    if format == .glsl {
        #expect(shader.glslTextureSamplerBindings.count == 4)
        #expect(Set(shader.glslTextureSamplerBindings.map(\.flatBindingIndex)).count == 4)
    }
    if format == .wgsl, case .text(_, let code) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(code))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func fragmentResourceParametersPreserveImplicitSampling(format: ShaderOutputFormat) throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var state: sampler;
        fn read(t: texture_2d<f32>, s: sampler, uv: vec2f) -> vec4f {
            return textureSample(t, s, uv);
        }
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            return read(image, state, uv);
        }
        """)
    let shader = try IndependentShaderCompiler().compileRenderStage(source, entryPoint: "fs", to: format)
    #expect(shader.sampledTextures.count == 1)
    #expect(shader.samplerBindings.count == 1)
    #expect(shader.sampledTextures.first?.requiresFiltering == true)
}

@Test func handlesRemainInvalidAsEntryInputsAndFunctionResults() {
    for source in [
        "@compute @workgroup_size(1) fn run(t: texture_2d<f32>) {}",
        "@group(0) @binding(0) var image: texture_2d<f32>; fn get() -> texture_2d<f32> { return image; }",
        "fn f(s: sampler_comparison) {} @group(0) @binding(0) var s: sampler; @compute @workgroup_size(1) fn run() { f(s); }",
    ] {
        #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
    }
}

#if canImport(Metal)
@Test func forwardedTextureSamplerAndStorageParametersRunOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let words = try await forwardedResourcesDiagnostic(device)
    let rgba = words.prefix(4).map { Int((Float(bitPattern: $0) * 255).rounded()) }
    #expect(rgba == [70, 10, 220, 150])
    #expect(Array(words.dropFirst(4)) == [0xff000046, 0xff0000dc])
    await device.close()
}

private func forwardedResourcesDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(forwardedResourceShader), to: .msl)
    let pipeline = try await device.createComputePipeline(shader)
    let sampleUsage: TextureUsage = [.textureBinding, .copyDestination]
    let storeUsage: TextureUsage = [.storageBinding, .copySource]
    let a = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: sampleUsage))
    let b = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: sampleUsage))
    let first = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: storeUsage))
    let second = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: storeUsage))
    let clamp = try await device.createSampler(.init())
    let repeating = try await device.createSampler(.init(addressModeU: .repeat))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let region = TextureCopyRegion(width: 2, height: 1)
    return try await device.submitCommands([
        .writeTexture(destination: a, data: [10, 0, 0, 255, 70, 0, 0, 255], region: region),
        .writeTexture(destination: b, data: [150, 0, 0, 255, 220, 0, 0, 255], region: region),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): a, .init(group: 1, binding: 0): b,
                                    .init(group: 2, binding: 0): first, .init(group: 2, binding: 1): second],
                         samplers: [.init(binding: 1): clamp, .init(group: 1, binding: 1): repeating]),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ]).wait().flatMap { $0 }
}
#endif
