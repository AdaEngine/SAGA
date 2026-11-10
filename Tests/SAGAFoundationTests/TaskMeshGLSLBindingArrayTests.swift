import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslTaskAndMeshSupportStaticBindingArrays() throws {
    let source = """
    enable wgpu_mesh_shader, wgpu_binding_array;
    struct Payload { value: u32, }
    struct Params { value: vec4f, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
    @group(1) @binding(0) var states: binding_array<sampler, 2>;
    @group(2) @binding(0) var<uniform> params: binding_array<Params, 2>;
    @group(3) @binding(0) var<storage, read_write> records: binding_array<Params, 2>;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u {
        let value = textureSampleLevel(images[1], states[0], vec2f(0.5), 0.0) + params[1].value;
        records[1].value = value;
        payload.value = u32(value.x);
        return vec3u(1u);
    }
    @mesh(out) @payload(payload) @workgroup_size(1)
    fn mesh() {
        let value = textureSampleLevel(images[1], states[0], vec2f(0.5), 0.0) + params[1].value;
        records[1].value = value;
        out.vertices[0].position = value;
        out.vertex_count = 3u; out.primitive_count = 1u; out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """
    let compiler = IndependentShaderCompiler()
    let artifacts = [try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl),
                     try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl)]
    for artifact in artifacts {
        guard case .text(.glsl, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(code.contains("uniform sampler2D saga_pair_"))
        #expect(code.contains("textureLod("))
        #expect(code.contains("saga_buffer_"))
    }
}
