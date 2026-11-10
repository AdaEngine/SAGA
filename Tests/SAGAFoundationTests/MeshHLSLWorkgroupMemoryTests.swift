import Testing
import SAGACompiler
import SAGAShaderIR

@Test func meshHLSLInitializesReachableWorkgroupGlobals() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: O;
    var<workgroup> scratch: array<u32, 7>;
    var<workgroup> counter: atomic<u32>;
    var<workgroup> unused: u32;
    fn fill(i: u32) { scratch[i] = atomicAdd(&counter, 1u); }
    @mesh(output) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
        fill(lane);
        workgroupBarrier();
        output.vertex_count = scratch[lane];
        output.primitive_count = 1u;
    }
    """
    let artifact = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .hlsl)
    guard case .text(_, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(code.contains("groupshared uint saga_shared_"))
    #expect(code.contains("_init < 7u"))
    #expect(code.contains("_init += 4u"))
    #expect(code.contains("InterlockedAdd("))
    #expect(code.contains("< 7u"))
    let lines = code.split(separator: "\n")
    #expect(lines.filter { $0.contains("groupshared uint saga_shared_") }.count == 2)
    let initEnd = try #require(lines.firstIndex { $0.contains("GroupMemoryBarrierWithGroupSync();") })
    let bodyCall = try #require(lines.firstIndex { $0.contains("    saga_mesh_body(") })
    #expect(initEnd < bodyCall)
}
