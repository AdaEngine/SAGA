import SAGACompiler
import SAGAShaderIR
import Testing

@Test(arguments: ["linear, centroid", "perspective, sample", "flat"])
func glslMeshPreservesVertexInterpolation(mode: String) throws {
    let source = """
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, @location(0) @interpolate(\(mode)) value: f32, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> out: O;
    @mesh(out) @workgroup_size(1) fn mesh() { out.vertex_count = 0u; out.primitive_count = 0u; }
    """
    guard case .text(_, let code) = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl) else { throw ShaderFailure.incompatibleArtifact }
    if mode == "linear, centroid" { #expect(code.contains("noperspective centroid out float")) }
    if mode == "perspective, sample" { #expect(code.contains("sample out float")) }
    if mode == "flat" { #expect(code.contains("flat out float")) }
}
