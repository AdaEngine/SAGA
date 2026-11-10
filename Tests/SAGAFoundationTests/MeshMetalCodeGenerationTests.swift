import Testing
import SAGABackends
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation

@Test func meshMetalStagesOutputAndCounts() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(4) fn main(@builtin(local_invocation_index) lane: u32) {
      if lane == 0u { output.vertex_count = 3u; output.primitive_count = 1u; output.vertices[0].position = vec4f(1.0); output.primitives[0].indices = vec3u(0u, 1u, 2u); }
    }
    """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.meshShader]))
    guard case .text(_, let text) = try MeshMetalCodeGenerator().generate(validated, entryPoint: "main") else { Issue.record("Expected text"); return }
    #expect(text.contains("metal::topology::triangle"))
    #expect(text.contains("set_vertex"))
    #expect(text.contains("set_index"))
    #expect(text.contains("set_primitive_count"))
    #expect(text.contains("saga_vertex_count > 0u"))
    #expect(text.contains("metal::min(saga_index_0, saga_vertex_count - 1u)"))
    #expect(text.contains("metal::threadgroup_barrier"))
}

@Test func meshMetalEmitsPrimitiveCulledAttribute() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, @builtin(cull_primitive) culled: bool, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(1) fn main() { output.primitives[0].culled = true; }
    """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.meshShader]))
    guard case .text(_, let text) = try MeshMetalCodeGenerator().generate(validated, entryPoint: "main") else { Issue.record("Expected text"); return }
    #expect(text.contains("[[primitive_culled]]"))
    #expect(!text.contains("[[cull_primitive]]"))
    #expect(text.contains("set_primitive(i, value)"))
}
