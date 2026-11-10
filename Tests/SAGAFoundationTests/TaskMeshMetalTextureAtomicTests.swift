import SAGACompiler
import SAGAShaderIR
import Testing

@Test func metalTaskAndMeshTextureAtomicsPreserveBindingsAndBounds() throws {
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @group(0) @binding(0) var image: texture_storage_2d_array<r32uint, atomic>;
    fn update() { textureAtomicAdd(image, vec2i(-1, 0), 1u, 3u); }
    @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
        update(); payload.value = 1u; return vec3u(1u);
    }
    @mesh(out) @payload(payload) @workgroup_size(1) fn mesh() {
        update(); out.vertex_count = 0u; out.primitive_count = 0u;
    }
    """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    for textures in [task.storageTextures, mesh.storageTextures] {
        #expect(textures.count == 1)
        #expect(textures[0].access.contains(.atomic))
        #expect(textures[0].isArrayed)
    }
    for artifact in [task.artifact, mesh.artifact] {
        guard case .text(.msl, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(code.contains("texture2d_array<uint, access::read_write>"))
        #expect(code.contains(".atomic_fetch_add("))
        #expect(code.contains(".get_array_size()"))
        #expect(code.contains(".get_width()"))
        #expect(!code.contains("saga_texture_atomic_guard"))
    }
}
