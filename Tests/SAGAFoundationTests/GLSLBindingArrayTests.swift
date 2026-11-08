import Testing
import SAGACompiler
import SAGAShaderIR

private let glslBindingArrayCompute = """
    enable wgpu_binding_array;
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
    @group(1) @binding(0) var states: binding_array<sampler, 2>;
    @group(2) @binding(1) var targets: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4f, 4>;
    fn sample(image: texture_2d<f32>, state: sampler) -> vec4f {
        return textureSampleLevel(image, state, vec2f(1.25, 0.5), 0.0);
    }
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let i = lane / 2u;
        let j = lane % 2u;
        let sampled = sample(images[i], states[j]);
        let color = sampled + textureLoad(images[i], vec2i(0), 0);
        output[lane] = color + vec4f(f32(textureDimensions(images[7]).x));
        if j == 0u { textureStore(targets[i], vec2u(0), color); }
    }
    """

@Test func glslBindingArraysReflectEveryTextureSamplerElementPair() throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(glslBindingArrayCompute), to: .glsl,
        bindingArrayLengths: [.init(group: 0, binding: 0): 2])
    #expect(shader.sampledTextures.first?.bindingArray?.count == 2)
    #expect(shader.storageTextures.first?.bindingArray?.count == 2)
    let pairs = shader.glslTextureSamplerBindings
    #expect(pairs.count == 4)
    #expect(Set(pairs.map(\.flatBindingIndex)).count == 4)
    #expect(pairs.allSatisfy { $0.texture == .init(group: 0, binding: 0) && $0.sampler == .init(group: 1, binding: 0) })
    #expect(Set(pairs.compactMap { pair in
        pair.textureArrayElement.flatMap { texture in pair.samplerArrayElement.map { texture * 2 + $0 } }
    }) == [0, 1, 2, 3])
    guard case .text(.glsl, let source) = shader.artifact else {
        Issue.record("Expected GLSL")
        return
    }
    #expect(source.split(separator: "\n").filter { $0.contains("uniform sampler2D saga_pair_") }.count == 4)
    #expect(source.split(separator: "\n").filter { $0.contains("uniform image2D") }.count == 2)
}

private let glslBindingArrayRender = """
    enable wgpu_binding_array;
    struct Offset { value: vec4f }
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
    @group(1) @binding(0) var states: binding_array<sampler, 2>;
    @group(2) @binding(1) var<uniform> offsets: binding_array<Offset, 2>;
    @vertex fn vs() -> @builtin(position) vec4f {
        let texel = textureSampleLevel(images[1], states[0], vec2f(0.5), 0.0);
        return vec4f(texel.xyz + offsets[0].value.xyz, 1.0);
    }
    @fragment fn fs(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
        let uv = pixel.xy + offsets[u32(pixel.x)].value.xy;
        return textureSample(images[u32(pixel.x)], states[u32(pixel.y)], uv);
    }
    """

@Test func glslBindingArrayPairsKeepSharedRenderUnits() throws {
    let input = ShaderInput.wgsl(glslBindingArrayRender)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .glsl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .glsl)
    let shared = try #require(vertex.glslTextureSamplerBindings.first)
    #expect(vertex.glslTextureSamplerBindings.count == 1)
    #expect(shared.textureArrayElement == 1)
    #expect(shared.samplerArrayElement == 0)
    #expect(fragment.glslTextureSamplerBindings.contains(shared))
    #expect(vertex.uniformBindings.first?.bindingArray?.count == 2)
    #expect(vertex.uniformBindings.first?.flatBindingIndex == fragment.uniformBindings.first?.flatBindingIndex)
    guard case .text(.glsl, let source) = fragment.artifact else {
        Issue.record("Expected GLSL")
        return
    }
    #expect(source.split(separator: "\n").filter { $0.contains("= texture(") }.count == 4)
}

private let glslBindingArrayBuffers = """
    enable wgpu_binding_array;
    requires unrestricted_pointer_parameters;
    struct Data { counter: atomic<u32>, values: array<u32> }
    struct Settings { value: vec4u }
    @group(0) @binding(0) var<storage, read_write> data: binding_array<Data>;
    @group(0) @binding(1) var<uniform> settings: binding_array<Settings, 2>;
    @group(0) @binding(2) var<storage, read_write> out: array<vec4u, 4>;
    fn update(p: ptr<storage, atomic<u32>, read_write>) -> u32 { return atomicAdd(p, 1u); }
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let i = i32(lane) - 1;
        let previous = update(&data[i].counter);
        let count = arrayLength(&data[i].values);
        out[lane] = vec4u(previous, count, data[i].values[0], settings[i].value.x);
        data[i].values[0] = settings[i].value.y;
        data[7].values[0] = 99u;
    }
    """

@Test func glslBindingArraysPreserveBufferRangesAndAtomicSelection() throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(glslBindingArrayBuffers), to: .glsl,
        bindingArrayLengths: [.init(group: 0, binding: 0): 2])
    let data = try #require(shader.storageBindings.first { $0.binding.binding == 0 })
    let settings = try #require(shader.uniformBindings.first)
    #expect(data.bindingArray?.count == 2)
    #expect(data.flatBindingIndex == 0)
    #expect(data.runtimeArrayStride == 4)
    #expect(data.runtimeArrayOffset == 4)
    #expect(settings.bindingArray?.count == 2)
    guard case .text(.glsl, let source) = shader.artifact else {
        Issue.record("Expected GLSL")
        return
    }
    #expect(source.split(separator: "\n").filter { $0.contains("atomicAdd(") }.count == 2)
    #expect(source.contains("saga_buffer_0_0.words.length()"))
    #expect(source.contains("saga_buffer_0_1.words.length()"))
    #expect(!source.contains("= 99u;"))
}

@Test func glslBindingArrayLayoutsIncludeOverridesFromLinkedStages() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        override fragmentCount = 3u;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
        @group(0) @binding(1) var extra: binding_array<texture_2d<f32>, fragmentCount>;
        @group(1) @binding(0) var state: sampler;
        @vertex fn vs() -> @builtin(position) vec4f {
            return textureSampleLevel(images[0], state, vec2f(0.5), 0.0);
        }
        @fragment fn fs(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
            return textureSample(images[0], state, pixel.xy) + textureSample(extra[u32(pixel.x)], state, pixel.xy);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .glsl, constants: ["fragmentCount": 4])
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .glsl, constants: ["fragmentCount": 4])
    let shared = try #require(vertex.glslTextureSamplerBindings.first)
    #expect(fragment.glslTextureSamplerBindings.contains(shared))
}
