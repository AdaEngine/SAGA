import Testing
import SAGACompiler

@Test func meshGLSLSubgroupsDeclareRequiredExtensions() throws {
    let source = """
    enable wgpu_mesh_shader;
    enable subgroups;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(32) fn main(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32) {
      let total = subgroupAdd(lane);
      if lane == 0u { output.vertex_count = 3u; output.primitive_count = 1u; output.vertices[0].position = vec4f(f32(total + size)); output.primitives[0].indices = vec3u(0u, 1u, 2u); }
    }
    """
    let shader = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "main", to: .glsl)
    guard case .text(_, let text) = shader else { Issue.record("Expected GLSL source"); return }
    #expect(text.contains("#extension GL_KHR_shader_subgroup_basic : require"))
    #expect(text.contains("#extension GL_KHR_shader_subgroup_arithmetic : require"))
    #expect(text.contains("gl_SubgroupInvocationID"))
    #expect(text.contains("gl_SubgroupSize"))
    #expect(text.contains("subgroupAdd("))
}
