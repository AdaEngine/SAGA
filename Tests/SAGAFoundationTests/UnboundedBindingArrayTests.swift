import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let layoutCounts: [IRResourceBinding: UInt32] = [
    .init(group: 0, binding: 0): 2,
    .init(group: 0, binding: 1): 1,
    .init(group: 0, binding: 2): 2,
    .init(group: 0, binding: 3): 2,
    .init(group: 1, binding: 0): 2,
    .init(group: 1, binding: 1): 2,
]

private let unboundedShader = """
    enable wgpu_binding_array;
    struct Record { tail: array<u32> }
    struct Settings { flags: vec4u }
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
    @group(0) @binding(1) var alternates: binding_array<texture_2d<f32>>;
    @group(0) @binding(2) var states: binding_array<sampler>;
    @group(0) @binding(3) var targets: binding_array<texture_storage_2d<rgba8unorm, write>>;
    @group(1) @binding(0) var<storage, read_write> records: binding_array<Record>;
    @group(1) @binding(1) var<uniform> settings: binding_array<Settings>;
    @group(1) @binding(2) var<storage, read_write> output: array<vec4u, 4>;
    @group(2) @binding(0) var unused: binding_array<texture_2d<f32>>;

    fn readImage(image: texture_2d<f32>, state: sampler) -> f32 {
        return textureSampleLevel(image, state, vec2f(1.25, 0.5), 0.0).x;
    }

    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let index = i32(lane) - 1;
        let red = readImage(images[index], states[index]);
        let width = textureDimensions(alternates[7]).x;
        output[lane] = vec4u(u32(round(red * 255.0)) + width,
            arrayLength(&records[index].tail) + arrayLength(&records[7].tail),
            settings[index].flags.x, records[index].tail[0]);
        records[index].tail[0] = settings[index].flags.y;
        records[7].tail[0] = 999u;
        if lane == 1u || lane == 2u {
            textureStore(targets[index], vec2u(0), vec4f(red, 0, 0, 1));
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .wgsl])
func unboundedBindingArraysUseCountsPerLogicalBinding(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(unboundedShader)
    let module = try compiler.parseAndValidate(input, capabilities: .allKnown).module
    #expect(module.globalVariables.values[0].type == module.globalVariables.values[1].type)
    let shader = try compiler.compileCompute(input, to: format, bindingArrayLengths: layoutCounts)
    #expect(shader.sampledTextures.map { $0.bindingArray?.count } == [2, 1])
    #expect(shader.samplerBindings.first?.bindingArray?.count == 2)
    #expect(shader.storageTextures.first?.bindingArray?.count == 2)
    #expect(shader.storageBindings.first?.bindingArray?.count == 2)
    #expect(shader.uniformBindings.first?.bindingArray?.count == 2)
    #expect(try compiler.compile(input, to: format, bindingArrayLengths: layoutCounts) == shader.artifact)
    if case .text(.wgsl, let source) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: .allKnown)
    }
}

@Test func unboundedBindingArraysRejectInvalidLayoutLengths() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(unboundedShader)
    let images = IRResourceBinding(group: 0, binding: 0)
    for lengths in [
        layoutCounts.filter { $0.key != images },
        layoutCounts.merging([images: 0], uniquingKeysWith: { _, new in new }),
        layoutCounts.merging([images: .max], uniquingKeysWith: { _, new in new }),
        layoutCounts.merging([.init(group: 1, binding: 2): 2], uniquingKeysWith: { _, new in new }),
        layoutCounts.merging([.init(group: 3, binding: 9): 2], uniquingKeysWith: { _, new in new }),
    ] {
        #expect(throws: ShaderFailure.self) {
            try compiler.compileCompute(input, to: .msl, bindingArrayLengths: lengths)
        }
    }
}

@Test func unboundedBindingArrayLayoutsFollowSelectedGraphicsEntryPoint() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(1) var common: binding_array<texture_2d<f32>>;
        @group(0) @binding(2) var fragmentImages: binding_array<texture_2d<f32>>;
        @vertex fn vs() -> @builtin(position) vec4f {
            return vec4f(f32(textureDimensions(common[0]).x), 0, 0, 1);
        }
        @fragment fn fs() -> @location(0) vec4f {
            return textureLoad(common[0], vec2i(0), 0) + textureLoad(fragmentImages[0], vec2i(0), 0);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let shared = IRResourceBinding(group: 0, binding: 1)
    let fragmentOnly = IRResourceBinding(group: 0, binding: 2)
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .hlsl, bindingArrayLengths: [shared: 2])
    #expect(throws: ShaderFailure.self) {
        try compiler.compileRenderStage(input, entryPoint: "fs", to: .hlsl, bindingArrayLengths: [shared: 2])
    }
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .hlsl,
                                                   bindingArrayLengths: [shared: 2, fragmentOnly: 1])
    #expect(vertex.sampledTextures.map { $0.bindingArray?.count } == [2])
    #expect(fragment.sampledTextures.map { $0.bindingArray?.count } == [2, 1])
    #expect(vertex.sampledTextures.first?.bindingArray == fragment.sampledTextures.first?.bindingArray)
}

#if canImport(Metal)
@Test func unboundedBindingArraysUseLayoutOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(unboundedShader), to: .msl,
                                                                bindingArrayLengths: layoutCounts)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let sample = TextureDescriptor(width: 2, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination])
    let target = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.storageBinding, .copySource])
    let a = try await device.createTexture(sample)
    let b = try await device.createTexture(sample)
    let first = try await device.createTexture(target)
    let second = try await device.createTexture(target)
    let clamp = try await device.createSampler(.init())
    let repeating = try await device.createSampler(.init(addressModeU: .repeat))
    let records = try await device.createBuffer(.init(size: 512, usage: [.storage, .copyDestination, .copySource]))
    let settings = try await device.createBuffer(.init(size: 512, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
    let region = TextureCopyRegion(width: 2, height: 1)
    let words = try await device.submitCommands([
        .writeTexture(destination: a, data: [10, 0, 0, 255, 70, 0, 0, 255], region: region),
        .writeTexture(destination: b, data: [150, 0, 0, 255, 220, 0, 0, 255], region: region),
        .write(buffer: records, offset: 0, words: [5]),
        .write(buffer: records, offset: 256, words: [7, 8, 9]),
        .write(buffer: settings, offset: 0, words: [11, 55, 0, 0]),
        .write(buffer: settings, offset: 256, words: [22, 77, 0, 0]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(group: 1, binding: 0, arrayElement: 0): .init(buffer: records, size: 4),
            .init(group: 1, binding: 0, arrayElement: 1): .init(buffer: records, offset: 256, size: 12),
            .init(group: 1, binding: 1, arrayElement: 0): .init(buffer: settings, size: 16),
            .init(group: 1, binding: 1, arrayElement: 1): .init(buffer: settings, offset: 256, size: 16),
            .init(group: 1, binding: 2): .init(buffer: output),
        ], workgroups: .init(x: 1), textures: [
            .init(binding: 0, arrayElement: 0): a, .init(binding: 0, arrayElement: 1): b,
            .init(binding: 1, arrayElement: 0): a,
            .init(binding: 3, arrayElement: 0): first, .init(binding: 3, arrayElement: 1): second,
        ], samplers: [.init(binding: 2, arrayElement: 0): clamp, .init(binding: 2, arrayElement: 1): repeating]),
        .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        .copy(source: records, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
        .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 12), .read(buffer: readback, byteCount: 12),
        .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ]).wait()
    #expect(words == [[72, 0, 0, 0, 72, 1, 11, 5, 152, 3, 22, 7, 152, 0, 0, 0],
                      [55], [77, 8, 9], [0xff000046], [0xff000096]])
    await device.close()
}
#endif
