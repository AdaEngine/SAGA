import SAGACompiler
import SAGAShaderIR
import Testing

@Test func hlslTaskAndMeshAcceptRuntimeBindingArrayLengths() throws {
    let source = """
    enable wgpu_mesh_shader, wgpu_binding_array;
    struct Payload { value: u32, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
      payload.value = textureDimensions(images[1u]).x; return vec3u(1u);
    }
    @mesh(out) @workgroup_size(1) fn mesh() {
      out.vertex_count = textureDimensions(images[1u]).x; out.primitive_count = 0u;
    }
    """
    let compiler = IndependentShaderCompiler()
    let counts: [IRResourceBinding: UInt32] = [.init(group: 0, binding: 0): 2]
    let artifacts = [try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .hlsl, bindingArrayLengths: counts),
                     try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .hlsl, bindingArrayLengths: counts)]
    for artifact in artifacts {
        guard case .text(.hlsl, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(code.contains("Texture2D<float4>"))
        #expect(code.contains("[2]"))
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .hlsl,
                                      bindingArrayLengths: [.init(group: 0, binding: 0): 0])
    }
}
