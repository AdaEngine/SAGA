import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes

private func textureShader(_ count: Int) -> String {
    let declarations = (0..<count).map { "@group(0) @binding(\($0)) var t\($0): texture_2d<f32>;" }.joined(separator: "\n")
    let reads = (0..<count).map { "textureLoad(t\($0), vec2i(0), 0).x" }.joined(separator: " + ")
    return declarations + "\n@fragment fn fs() -> @location(0) vec4f { return vec4f(\(reads)); }"
}

@Test func renderSampledTextureProfileBoundaries() throws {
    let compiler = IndependentShaderCompiler()
    let mixed = """
        @group(0) @binding(99) var sampled: texture_2d<f32>;
        @group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
        @fragment fn fs() -> @location(0) vec4f {
            let value = textureLoad(sampled, vec2i(0), 0);
            textureStore(output, vec2i(0), value);
            return value;
        }
        """
    let profiled = try compiler.compileMetalRenderStage(.wgsl(mixed), entryPoint: "fs", maxSampledTextures: 1)
    let standard = try compiler.compileRenderStage(.wgsl(mixed), entryPoint: "fs", to: .msl)
    #expect(profiled.sampledTextures.first?.flatBindingIndex == 0)
    #expect(profiled.storageTextures.first?.flatBindingIndex == standard.storageTextures.first?.flatBindingIndex)
    #expect(profiled.storageTextures.first?.flatBindingIndex == 24)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalRenderStage(.wgsl(mixed), entryPoint: "fs", maxSampledTextures: 0)
    }
    let sixteen = try compiler.compileMetalRenderStage(.wgsl(textureShader(16)), entryPoint: "fs", maxSampledTextures: 16)
    #expect(sixteen.sampledTextures.count == 16)
    #expect(Set(sixteen.sampledTextures.map(\.flatBindingIndex)) == Set(0..<16))
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalRenderStage(.wgsl(textureShader(17)), entryPoint: "fs", maxSampledTextures: 16)
    }
    let seventeen = try compiler.compileMetalRenderStage(.wgsl(textureShader(17)), entryPoint: "fs", maxSampledTextures: 17)
    #expect(seventeen.sampledTextures.count == 17)
    #expect(Set(seventeen.sampledTextures.map(\.flatBindingIndex)) == Set(0..<17))
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalRenderStage(.wgsl(textureShader(18)), entryPoint: "fs", maxSampledTextures: 17)
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalRenderStage(.wgsl(textureShader(1)), entryPoint: "fs", maxSampledTextures: 25)
    }
}

@Test func metalProfileZeroAndLegacyArtifactCompatibility() throws {
    let compiler = IndependentShaderCompiler()
    let empty = ShaderInput.wgsl("@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }")
    let zero = try compiler.compileMetalRenderStage(empty, entryPoint: "fs", maxSampledTextures: 0)
    #expect(zero.sampledTextures.isEmpty)
    let full = ShaderInput.wgsl(textureShader(24))
    let profiled = try compiler.compileMetalRenderStage(full, entryPoint: "fs", maxSampledTextures: 24)
    let legacy = try compiler.compileRenderStage(full, entryPoint: "fs", to: .msl)
    #expect(profiled.sampledTextures.count == 24)
    #expect(Set(profiled.sampledTextures.map(\.flatBindingIndex)) == Set(0..<24))
    #expect(profiled == legacy)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalRenderStage(.wgsl(textureShader(25)), entryPoint: "fs", maxSampledTextures: 24)
    }
}

@Test func metalVertexPullingHonorsSampledTextureCeiling() throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(7) var image: texture_2d<f32>;
        @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f {
            let value = textureLoad(image, vec2i(0), 0).x;
            return vec4f(position, value, 1.0);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalVertexStage(source, entryPoint: "vs", vertexBuffers: layouts, maxSampledTextures: 0)
    }
    let shader = try compiler.compileMetalVertexStage(source, entryPoint: "vs", vertexBuffers: layouts, maxSampledTextures: 1)
    #expect(shader.sampledTextures.first?.flatBindingIndex == 0)
}

@Test func metalExternalTextureConsumesThreeProfileSlots() throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var frame: texture_external;
        @group(0) @binding(1) var state: sampler;
        @fragment fn fs() -> @location(0) vec4f {
            return textureSampleBaseClampToEdge(frame, state, vec2f(0.5));
        }
        """)
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalRenderStage(source, entryPoint: "fs", maxSampledTextures: 2)
    }
    let profiled = try compiler.compileMetalRenderStage(source, entryPoint: "fs", maxSampledTextures: 3)
    let legacy = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    #expect(profiled.metalSampledTextureSlotCount == 3)
    #expect(profiled == legacy)
}

@Test func metalSlotCountDoesNotClaimNonMetalLayouts() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(textureShader(1))
    let metal = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let wgsl = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    #expect(metal.metalSampledTextureSlotCount == 1)
    #expect(wgsl.metalSampledTextureSlotCount == nil)
}

@Test func metalMixedExternalArrayPerStageLayout() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(7) var images: binding_array<texture_2d<f32>, 2>;
        @group(1) @binding(1) var frame: texture_external;
        @group(1) @binding(2) var state: sampler;
        @vertex fn vs() -> @builtin(position) vec4f {
            return textureLoad(images[0], vec2i(0), 0);
        }
        @fragment fn fs() -> @location(0) vec4f {
            return textureLoad(images[1], vec2i(0), 0) + textureSampleBaseClampToEdge(frame, state, vec2f(0.5));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileMetalRenderStage(source, entryPoint: "vs", maxSampledTextures: 2)
    #expect(vertex.metalSampledTextureSlotCount == 2)
    #expect(vertex.externalTextureLayouts.isEmpty)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalRenderStage(source, entryPoint: "fs", maxSampledTextures: 4)
    }
    let fragment = try compiler.compileMetalRenderStage(source, entryPoint: "fs", maxSampledTextures: 5)
    #expect(fragment.metalSampledTextureSlotCount == 5)
    let array = try #require(fragment.sampledTextures.first)
    let base = try #require(array.flatBindingIndex)
    #expect(array.bindingArray?.count == 2)
    let external = try #require(fragment.externalTextureLayouts.first)
    #expect(external.planeBindingIndices.count == 3)
    let slots = [base, base + 1] + external.planeBindingIndices
    #expect(Set(slots) == Set(0..<5))
}

@Test func metalProfileBindingArrayBoundary() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(7) var images: binding_array<texture_2d<f32>, 2>;
        @group(1) @binding(1) var other: texture_2d<f32>;
        @fragment fn fs() -> @location(0) vec4f {
            return textureLoad(images[0], vec2i(0), 0) + textureLoad(images[1], vec2i(0), 0) + textureLoad(other, vec2i(0), 0);
        }
        """)
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalRenderStage(source, entryPoint: "fs", maxSampledTextures: 2)
    }
    let shader = try compiler.compileMetalRenderStage(source, entryPoint: "fs", maxSampledTextures: 3)
    #expect(shader.metalSampledTextureSlotCount == 3)
    #expect(Set(shader.sampledTextures.map(\.flatBindingIndex)) == Set([0, 2]))
}
