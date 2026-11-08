import Testing
import SAGACompiler
import SAGAShaderIR

@Test func webBindingArraysLowerToStandardWGSL() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
        @group(0) @binding(1) var states: binding_array<sampler, 2>;
        @group(0) @binding(2) var targets: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
        @group(0) @binding(999) var<storage, read_write> values: array<vec4f, 2>;
        @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) i: u32) {
            let sampled = textureSampleLevel(images[i], states[1u - i], vec2f(0.5), 0.0);
            let texel = textureLoad(images[i], vec2i(0), 0);
            let dimensions = textureDimensions(images[i]);
            let color = sampled + texel + vec4f(f32(dimensions.x));
            textureStore(targets[i], vec2u(0), color);
            values[i] = color;
        }
        """)
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileWebGPUCompute(input, bindingArrayLengths: [.init(group: 0, binding: 0): 2])
    guard case .text(.wgsl, let source) = artifact.artifact else {
        Issue.record("Expected WGSL")
        return
    }
    let layouts = [artifact.sampledTextures.first?.bindingArray, artifact.samplerBindings.first?.bindingArray,
                   artifact.storageTextures.first?.bindingArray]
    let bases = try layouts.map { try #require($0?.webBindingBase) }
    let slots = bases.flatMap { [$0, $0 + 1] }
    #expect(Set(slots).count == 6)
    #expect(slots.allSatisfy { $0 < 999 })
    let module = try compiler.parseAndValidate(.wgsl(source)).module
    #expect(module.globalVariables.values.compactMap(\.binding).count == 7)
    #expect(!source.contains("wgpu_binding_array"))
}

@Test func webBindingArraySamplingKeepsImplicitDerivativesUniform() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
        @group(0) @binding(1) var states: binding_array<sampler, 2>;
        fn sample(t: texture_2d<f32>, s: sampler, uv: vec2f) -> vec4f { return textureSample(t, s, uv); }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            return vec4f(f32(textureDimensions(images[i]).x), 0, 0, 1);
        }
        @fragment fn fs(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
            return sample(images[u32(pixel.x)], states[u32(pixel.y)], pixel.xy);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileWebGPURenderStage(input, entryPoint: "vs")
    let fragment = try compiler.compileWebGPURenderStage(input, entryPoint: "fs")
    let base = try #require(vertex.sampledTextures.first?.bindingArray?.webBindingBase)
    #expect(fragment.sampledTextures.first?.bindingArray?.webBindingBase == base)
    guard case .text(.wgsl, let source) = fragment.artifact else {
        Issue.record("Expected WGSL")
        return
    }
    _ = try compiler.parseAndValidate(.wgsl(source))
    #expect(source.split(separator: "\n").filter { $0.contains("textureSample(") }.count == 4)
}
