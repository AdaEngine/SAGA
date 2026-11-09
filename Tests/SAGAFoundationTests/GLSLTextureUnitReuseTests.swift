import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslTextureFetchAndQueryReuseSampledTextureUnits() throws {
    let declarations = (0..<9).map { index in
        "@group(0) @binding(\(index)) var texture\(index): texture_2d<f32>;"
    }.joined(separator: "\n")
    let reads = (0..<9).map { index in
        """
        total += textureSampleLevel(texture\(index), state, vec2f(0.5), 0.0).x;
        total += textureLoad(texture\(index), vec2u(0), 0).x;
        total += f32(textureDimensions(texture\(index), 0).x);
        """
    }.joined(separator: "\n")
    let input = ShaderInput.wgsl("""
        \(declarations)
        @group(0) @binding(9) var state: sampler;
        @group(0) @binding(10) var<storage, read_write> output: f32;
        @compute @workgroup_size(1)
        fn main() {
            var total = 0.0;
            \(reads)
            output = total;
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .glsl)
    guard case .text(.glsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(shader.glslTextureSamplerBindings.count == 9)
    #expect(source.split(separator: "uniform sampler2D ").count == 10)
    #expect(source.split(separator: "texelFetch(saga_pair_").count == 10)
    for pair in shader.glslTextureSamplerBindings {
        #expect(shader.sampledTextures.first { $0.binding == pair.texture }?.flatBindingIndex == pair.flatBindingIndex)
    }
}

@Test(arguments: [false, true])
func glslShadowTextureQueriesReuseThePairUnlessTheTextureIsAlsoFetched(_ fetch: Bool) throws {
    let input = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_depth_2d;
        @group(0) @binding(1) var state: sampler_comparison;
        @fragment fn main() -> @location(0) vec4f {
            let compare = textureSampleCompare(image, state, vec2f(0.5), 0.5);
            let size = f32(textureDimensions(image, 0).x);
            let raw = \(fetch ? "textureLoad(image, vec2u(0), 0)" : "0.0");
            return vec4f(compare, size, raw, 1.0);
        }
        """)
    let shader = try IndependentShaderCompiler().compileRenderStage(input, entryPoint: "main", to: .glsl)
    guard case .text(.glsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    if fetch {
        #expect(source.contains("uniform sampler2D saga_texture_"))
        #expect(source.contains("texelFetch(saga_texture_"))
    } else {
        #expect(!source.contains("uniform sampler2D saga_texture_"))
        #expect(source.contains("textureSize(saga_pair_"))
    }
    #expect(source.contains("uniform sampler2DShadow saga_pair_"))
}
