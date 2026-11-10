import SAGACompiler
import SAGAShaderIR
import Testing

@Test func hlslTaskAndMeshSupportBindingArrays() throws {
    func shader(_ count: String) -> String { """
    enable wgpu_mesh_shader, wgpu_binding_array;
    struct Payload { value: u32, }
    struct Params { value: vec4f, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>\(count)>;
    @group(0) @binding(1) var targets: binding_array<texture_storage_2d<rgba8unorm, write>\(count)>;
    @group(1) @binding(0) var states: binding_array<sampler\(count)>;
    @group(2) @binding(0) var<uniform> params: binding_array<Params\(count)>;
    @group(3) @binding(0) var<storage, read_write> records: binding_array<Params\(count)>;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u {
        let value = textureSampleLevel(images[1], states[0], vec2f(0.5), 0.0) + params[1].value;
        records[1].value = value;
        textureStore(targets[1], vec2u(0), value);
        payload.value = u32(value.x);
        return vec3u(1u);
    }
    @mesh(out) @payload(payload) @workgroup_size(1)
    fn mesh() {
        let value = textureSampleLevel(images[1], states[0], vec2f(0.5), 0.0) + params[1].value;
        records[1].value = value;
        textureStore(targets[1], vec2u(0), value);
        out.vertices[0].position = value;
        out.vertex_count = 3u; out.primitive_count = 1u; out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """ }
    let compiler = IndependentShaderCompiler()
    for dynamic in [false, true] {
        let source = shader(dynamic ? "" : ", 2")
        let lengths: [IRResourceBinding: UInt32] = dynamic
            ? Dictionary(uniqueKeysWithValues: (0...3).map { (.init(group: UInt32($0), binding: 0), 2) } + [(.init(group: 0, binding: 1), 2)]) : [:]
        let artifacts = [try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .hlsl, bindingArrayLengths: lengths),
                         try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .hlsl, bindingArrayLengths: lengths)]
        for artifact in artifacts {
            guard case .text(.hlsl, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
            #expect(code.contains("Texture2D<float4> saga_texture_"))
            #expect(code.contains("SamplerState saga_sampler_"))
            #expect(code.contains("[2] : register(t"))
            #expect(code.contains("[2] : register(s"))
            #expect(code.contains("[2] : register(b"))
            #expect(code.contains("[2] : register(u"))
            #expect(code.contains("RWTexture2D<unorm float4> saga_texture_"))
        }
    }
}
