import Testing
import SAGACompiler

@Test func meshGLSLTextureBindingsEmitCombinedSamplersAndImages() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> out: O;
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var smp: sampler;
    @group(0) @binding(2) var target: texture_storage_2d<rgba8unorm, write>;
    fn sample() -> vec4f { return textureSampleLevel(image, smp, vec2f(0.5), 0.0); }
    @mesh(out) @workgroup_size(1) fn main() {
      let color = sample(); textureStore(target, vec2i(0), color);
      out.vertices[0].position = color;
      out.vertex_count = 3u; out.primitive_count = 1u; out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """
    let result = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "main", to: .glsl)
    guard case .text(_, let text) = result else { Issue.record("Expected GLSL source"); return }
    #expect(text.contains("uniform sampler2D saga_pair_"))
    #expect(text.contains("writeonly uniform image2D saga_texture_"))
    #expect(text.contains("textureLod("))
    #expect(text.contains("imageStore("))
}
