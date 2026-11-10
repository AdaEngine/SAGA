import SAGACompiler
import SAGAShaderIR
import Testing

@Test func hlslMeshEmitsSubgroupOperationsAndBuiltins() throws {
    let source = """
    enable wgpu_mesh_shader, subgroups;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> out: O;
    @mesh(out) @workgroup_size(32) fn mesh(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32) {
      let sum = subgroupAdd(lane);
      if lane == 0u { out.vertices[0].position = vec4f(f32(sum + size)); out.vertex_count = 1u; out.primitive_count = 0u; }
    }
    """
    let artifact = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .hlsl)
    guard case .text(_, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(code.contains("WaveGetLaneIndex()"))
    #expect(code.contains("WaveGetLaneCount()"))
    #expect(code.contains("WaveActiveSum("))
}

@Test func hlslMeshInitializesSubgroupIndices() throws {
    let source = """
    requires subgroup_id;
    enable wgpu_mesh_shader, subgroups;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> out: O;
    @mesh(out) @workgroup_size(32) fn mesh(@builtin(subgroup_id) group: u32, @builtin(num_subgroups) count: u32) {
      out.vertex_count = group + count; out.primitive_count = 0u;
    }
    """
    let artifact = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .hlsl)
    guard case .text(_, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(code.contains("groupshared uint saga_subgroup_count;"))
    #expect(code.contains("InterlockedAdd(saga_subgroup_count"))
    #expect(code.contains("WaveReadLaneFirst("))
}
