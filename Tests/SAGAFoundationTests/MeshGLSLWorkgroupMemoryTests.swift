import Testing
import SAGACompiler
import SAGAShaderIR

@Test func meshGLSLInitializesReachableWorkgroupGlobals() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: O;
    var<workgroup> scratch: array<u32, 7>;
    var<workgroup> counter: atomic<u32>;
    fn fill() { scratch[0] = atomicAdd(&counter, 1u); }
    @mesh(output) @workgroup_size(4) fn mesh() {
        fill();
        workgroupBarrier();
        output.vertex_count = scratch[0];
        output.primitive_count = 1u;
        output.vertices[0].position = vec4f(0.0);
        output.primitives[0].indices = vec3u(0u);
    }
    """
    let artifact = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl)
    guard case .text(_, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(code.contains("shared uint saga_shared_"))
    #expect(code.contains("_init < 7u"))
    #expect(code.contains("_init += 4u"))
    #expect(code.contains("atomicAdd("))
    #expect(code.contains("barrier();"))
}
