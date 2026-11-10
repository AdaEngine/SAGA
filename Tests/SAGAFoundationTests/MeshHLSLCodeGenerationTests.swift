import Testing
import SAGABackends
import SAGACompiler
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation

@Test func hlslMeshWritesOutputAndIndices() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Vertex { @builtin(position) position: vec4f, @location(0) color: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<Vertex, 3>, @builtin(primitives) primitives: array<Primitive, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(4) fn main(@builtin(local_invocation_index) lane: u32) {
        if lane == 0u { output.vertex_count = 3u; output.primitive_count = 1u; output.vertices[0].position = vec4f(0.0, 0.0, 0.0, 1.0); output.primitives[0].indices = vec3u(0u, 1u, 2u); }
    }
    """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.meshShader]))
    guard case .text(_, let hlsl) = try MeshHLSLCodeGenerator().generate(validated, entryPoint: "main") else { Issue.record("Expected HLSL"); return }
    #expect(hlsl.contains("saga_vertex_count == 0u ? 0u"))
    #expect(hlsl.contains("SetMeshOutputCounts(saga_vertex_count, saga_primitive_count)"))
    #expect(hlsl.contains("(uint3)(saga_vertex_count - 1u)"))
    #expect(hlsl.contains("out indices uint3"))
    #expect(hlsl.contains("SV_Position"))
    #expect(hlsl.contains("TEXCOORD0"))
    #expect(hlsl.contains("GroupMemoryBarrierWithGroupSync"))
}

@Test func hlslMeshWritesCullPrimitive() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, @builtin(cull_primitive) culled: bool, }
    struct Output { @builtin(vertices) vertices: array<Vertex, 3>, @builtin(primitives) primitives: array<Primitive, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(1) fn main() {
        output.vertex_count = 3u;
        output.primitive_count = 1u;
        output.primitives[0].culled = true;
    }
    """
    guard case .text(_, let hlsl) = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "main", to: .hlsl) else { Issue.record("Expected HLSL"); return }
    #expect(hlsl.contains("bool saga_field_1 : SV_CullPrimitive;"))
    #expect(hlsl.contains("saga_primitives[saga_primitive_index].saga_field_1 ="))
}

@Test func hlslMeshAndTaskUseBoundBuffers() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Config { value: u32, }
    struct Data { values: array<u32, 4>, }
    struct Payload { value: u32, }
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<Vertex, 3>, @builtin(primitives) primitives: array<Primitive, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    @group(2) @binding(3) var<uniform> config: Config;
    @group(1) @binding(4) var<storage, read_write> data: Data;
    var<task_payload> payload: Payload;
    var<workgroup> output: Output;
    @task @payload(payload) @workgroup_size(1) fn task_main() -> @builtin(mesh_task_size) vec3u {
        payload.value = data.values[config.value];
        return vec3u(1);
    }
    @mesh(output) @payload(payload) @workgroup_size(1) fn mesh_main() {
        data.values[config.value] = payload.value;
        output.vertex_count = data.values[config.value];
    }
    """
    let compiler = IndependentShaderCompiler()
    for entry in ["task_main", "mesh_main"] {
        let artifact = try entry == "task_main" ? compiler.compileTaskStage(.wgsl(source), entryPoint: entry, to: .hlsl) : compiler.compileMeshStage(.wgsl(source), entryPoint: entry, to: .hlsl)
        guard case .text(_, let hlsl) = artifact else { Issue.record("Expected HLSL"); continue }
        #expect(hlsl.contains("register(b3, space2)"))
        #expect(hlsl.contains("register(u4, space1)"))
        #expect(hlsl.contains("< 4u"))
    }
}

@Test func hlslMeshRejectsWorkgroupsAbove128Threads() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<Vertex, 3>, @builtin(primitives) primitives: array<Primitive, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(129) fn main() {}
    """
    #expect(throws: ShaderFailure.unsupported("HLSL mesh workgroup size exceeds numthreads limits")) {
        try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "main", to: .hlsl)
    }
}
