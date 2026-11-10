import SAGACompiler
import SAGAShaderIR
import Testing

@Test func metalTaskAndMeshStorageTextureShapes() throws {
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @group(0) @binding(0) var line: texture_storage_1d<r32uint, read_write>;
    @group(0) @binding(1) var layers: texture_storage_2d_array<r32uint, read_write>;
    @group(0) @binding(2) var volume: texture_storage_3d<r32uint, read_write>;
    fn update() -> u32 {
        let a = textureLoad(line, 0u);
        let b = textureLoad(layers, vec2u(0), 0u);
        let c = textureLoad(volume, vec3u(0));
        textureStore(line, 0u, a);
        textureStore(layers, vec2u(0), 0u, b);
        textureStore(volume, vec3u(0), c);
        return textureDimensions(line) + textureNumLayers(layers) + textureDimensions(volume).z;
    }
    @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
        payload.value = update(); return vec3u(1u);
    }
    @mesh(out) @payload(payload) @workgroup_size(1) fn mesh() {
        let value = update() + payload.value;
        out.vertices[0].position = vec4f(f32(value));
        out.vertex_count = 3u; out.primitive_count = 1u; out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    for textures in [task.storageTextures, mesh.storageTextures] {
        #expect(textures.map(\.dimension) == [.d1, .d2, .d3])
        #expect(textures.map(\.isArrayed) == [false, true, false])
        #expect(textures.allSatisfy { $0.access == [.load, .store] })
    }
    for artifact in [task.artifact, mesh.artifact] {
        guard case .text(.msl, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(code.contains("texture1d<uint, access::read_write>"))
        #expect(code.contains("texture2d_array<uint, access::read_write>"))
        #expect(code.contains("texture3d<uint, access::read_write>"))
        #expect(code.contains(".get_array_size()"))
        #expect(code.contains(".get_depth()"))
    }
}
