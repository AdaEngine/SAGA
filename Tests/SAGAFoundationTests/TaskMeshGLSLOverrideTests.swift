import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslTaskMeshOverridesKeepLinkedLayoutsAndIgnoreOtherStages() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_mesh_shader, wgpu_binding_array;
        override count: u32;
        override vertexCount: u32;
        override taskValue: u32;
        override meshValue: u32;
        struct Data { value: u32, }
        struct Payload { value: u32, }
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        @group(0) @binding(0) var<storage, read> task_data: binding_array<Data, count>;
        @group(0) @binding(1) var<storage, read> shared_data: Data;
        @group(1) @binding(0) var<storage, read> vertex_data: binding_array<Data, vertexCount>;
        var<task_payload> payload: Payload;
        var<workgroup> out: O;
        @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
            payload.value = task_data[0].value + shared_data.value + taskValue;
            return vec3u(1u);
        }
        @mesh(out) @payload(payload) @workgroup_size(1) fn mesh() {
            out.vertex_count = payload.value + shared_data.value + meshValue;
            out.primitive_count = 0u;
        }
        @vertex fn vertex() -> @builtin(position) vec4f {
            return vec4f(f32(vertex_data[0].value));
        }
        """)
    let compiler = IndependentShaderCompiler()
    guard case .text(.glsl, let task) = try compiler.compileTaskStage(
        input, entryPoint: "task", to: .glsl, constants: ["count": 3, "taskValue": 1]),
        case .text(.glsl, let mesh) = try compiler.compileMeshStage(
        input, entryPoint: "mesh", to: .glsl, constants: ["count": 3, "meshValue": 1]) else {
        throw ShaderFailure.incompatibleArtifact
    }
    #expect(task.contains("layout(std430, binding = 3) readonly buffer"))
    #expect(mesh.contains("layout(std430, binding = 3) readonly buffer"))
}
