import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslTaskAndMeshShareAggregateBufferRepresentation() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Data { value: u32, }
    struct Payload { value: u32, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    @group(0) @binding(0) var<storage, read> data: Data;
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
        payload.value = data.value;
        return vec3u(1u);
    }
    @mesh(out) @payload(payload) @workgroup_size(1) fn mesh() {
        let whole = data;
        out.vertex_count = whole.value + payload.value;
        out.primitive_count = 0u;
    }
    """
    let compiler = IndependentShaderCompiler()
    guard case .text(.glsl, let task) = try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl),
          case .text(.glsl, let mesh) = try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let taskBlock = try #require(task.split(separator: "\n").first { $0.contains("} saga_buffer_0;") })
    let meshBlock = try #require(mesh.split(separator: "\n").first { $0.contains("} saga_buffer_0;") })
    #expect(taskBlock == meshBlock)
}

@Test func glslTaskAndMeshShareBufferBindingUnits() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Data { value: u32, }
    struct Payload { value: u32, }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    @group(1) @binding(0) var<uniform> task_uniform: Data;
    @group(1) @binding(1) var<uniform> shared_uniform: Data;
    @group(1) @binding(2) var<uniform> mesh_uniform: Data;
    @group(2) @binding(0) var<storage, read> task_storage: Data;
    @group(2) @binding(1) var<storage, read> shared_storage: Data;
    @group(2) @binding(2) var<storage, read> mesh_storage: Data;
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
        payload.value = task_uniform.value + shared_uniform.value + task_storage.value + shared_storage.value;
        return vec3u(1u);
    }
    @mesh(out) @payload(payload) @workgroup_size(1) fn mesh() {
        out.vertex_count = payload.value + shared_uniform.value + mesh_uniform.value + shared_storage.value + mesh_storage.value;
        out.primitive_count = 0u;
    }
    """
    let compiler = IndependentShaderCompiler()
    guard case .text(.glsl, let task) = try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl),
          case .text(.glsl, let mesh) = try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl) else {
        throw ShaderFailure.incompatibleArtifact
    }
    func binding(_ code: String, buffer: Int) -> UInt32? {
        guard let line = code.split(separator: "\n").first(where: { $0.contains("} saga_buffer_\(buffer);") }),
              let start = line.range(of: "binding = "), let end = line[start.upperBound...].firstIndex(of: ")") else { return nil }
        return UInt32(line[start.upperBound..<end])
    }
    for (shared, taskOnly, meshOnly) in [(1, 0, 2), (4, 3, 5)] {
        let taskShared = binding(task, buffer: shared), meshShared = binding(mesh, buffer: shared)
        let taskUnit = binding(task, buffer: taskOnly), meshUnit = binding(mesh, buffer: meshOnly)
        #expect(taskShared != nil && taskShared == meshShared)
        #expect(taskUnit != nil && meshUnit != nil && taskUnit != meshUnit)
        #expect(taskUnit != taskShared && meshUnit != meshShared)
    }
}

@Test func glslTaskAndMeshReserveLinkedBufferArrayElements() throws {
    for extent in ["2", ""] {
        let source = """
        enable wgpu_mesh_shader, wgpu_binding_array;
        struct Data { value: u32, }
        struct Payload { value: u32, }
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        @group(1) @binding(0) var<storage, read> task_data: binding_array<Data\(extent.isEmpty ? "" : ", \(extent)")>;
        @group(1) @binding(1) var<storage, read> shared_data: Data;
        @group(1) @binding(2) var<storage, read> mesh_data: Data;
        var<task_payload> payload: Payload;
        var<workgroup> out: O;
        @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
            payload.value = task_data[1].value + shared_data.value;
            return vec3u(1u);
        }
        @mesh(out) @payload(payload) @workgroup_size(1) fn mesh() {
            out.vertex_count = payload.value + shared_data.value + mesh_data.value;
            out.primitive_count = 0u;
        }
        """
        let compiler = IndependentShaderCompiler()
        let lengths: [IRResourceBinding: UInt32] = extent.isEmpty ? [.init(group: 1, binding: 0): 2] : [:]
        guard case .text(.glsl, let task) = try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl, bindingArrayLengths: lengths),
              case .text(.glsl, let mesh) = try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl, bindingArrayLengths: lengths) else {
            throw ShaderFailure.incompatibleArtifact
        }
        #expect(task.contains("layout(std430, binding = 0) readonly buffer SagaStorageArray0_0"))
        #expect(task.contains("layout(std430, binding = 1) readonly buffer SagaStorageArray0_1"))
        #expect(task.contains("layout(std430, binding = 2) readonly buffer"))
        #expect(mesh.contains("layout(std430, binding = 2) readonly buffer"))
        #expect(mesh.contains("layout(std430, binding = 3) readonly buffer"))
    }
}
