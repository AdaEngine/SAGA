import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslTaskAndMeshReserveCombinedTextureUnitsAcrossStages() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<Vertex, 3>, @builtin(primitives) primitives: array<Primitive, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> output: Output;
    @group(0) @binding(0) var common: texture_2d<f32>;
    @group(0) @binding(1) var task_image: texture_2d<f32>;
    @group(0) @binding(2) var mesh_image: texture_2d<f32>;
    @group(0) @binding(3) var state: sampler;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u {
        let a = textureSampleLevel(common, state, vec2f(0.5), 0.0);
        let b = textureSampleLevel(task_image, state, vec2f(0.5), 0.0);
        payload.value = u32(a.x + b.x);
        return vec3u(1u);
    }
    @mesh(output) @payload(payload) @workgroup_size(1)
    fn mesh() {
        let a = textureSampleLevel(common, state, vec2f(0.5), 0.0);
        let b = textureSampleLevel(mesh_image, state, vec2f(0.5), 0.0);
        output.vertices[0].position = a + b;
        output.vertex_count = payload.value;
        output.primitive_count = 0u;
    }
    """
    let compiler = IndependentShaderCompiler()
    guard case .text(.glsl, let task) = try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl),
          case .text(.glsl, let mesh) = try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl) else {
        throw ShaderFailure.incompatibleArtifact
    }
    func combinedUnits(_ code: String) -> [String: UInt32] {
        var result: [String: UInt32] = [:]
        for line in code.split(separator: "\n") where line.contains("uniform sampler2D saga_pair_") {
            guard let start = line.range(of: "binding="), let end = line[start.upperBound...].firstIndex(of: ")"),
                  let unit = UInt32(line[start.upperBound..<end]), let last = line.split(separator: " ").last else { continue }
            result[String(last.dropLast())] = unit
        }
        return result
    }
    let taskUnits = combinedUnits(task), meshUnits = combinedUnits(mesh)
    let shared = Set(taskUnits.keys).intersection(meshUnits.keys)
    let taskOnly = Set(taskUnits.keys).subtracting(shared), meshOnly = Set(meshUnits.keys).subtracting(shared)
    #expect(shared.count == 1)
    #expect(taskOnly.count == 1)
    #expect(meshOnly.count == 1)
    for name in shared { #expect(taskUnits[name] == meshUnits[name]) }
    for a in taskOnly { for b in meshOnly { #expect(taskUnits[a] != meshUnits[b]) } }
}

@Test func glslTaskAndMeshReserveLinkedBindingArrayElements() throws {
    let source = """
    enable wgpu_mesh_shader, wgpu_binding_array;
    struct Payload { value: u32, }
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<Vertex, 3>, @builtin(primitives) primitives: array<Primitive, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<task_payload> payload: Payload;
    var<workgroup> output: Output;
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 3>;
    @group(0) @binding(1) var state: sampler;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u {
        payload.value = u32(textureSampleLevel(images[1], state, vec2f(0.5), 0.0).x);
        return vec3u(1u);
    }
    @mesh(output) @payload(payload) @workgroup_size(1)
    fn mesh() {
        output.vertices[0].position = textureSampleLevel(images[2], state, vec2f(0.5), 0.0);
        output.vertex_count = payload.value;
        output.primitive_count = 0u;
    }
    """
    let compiler = IndependentShaderCompiler()
    guard case .text(.glsl, let task) = try compiler.compileTaskStage(.wgsl(source), entryPoint: "task", to: .glsl),
          case .text(.glsl, let mesh) = try compiler.compileMeshStage(.wgsl(source), entryPoint: "mesh", to: .glsl) else {
        throw ShaderFailure.incompatibleArtifact
    }
    func combinedUnit(_ code: String) -> UInt32? {
        guard let line = code.split(separator: "\n").first(where: { $0.contains("uniform sampler2D saga_pair_") }),
              let start = line.range(of: "binding="), let end = line[start.upperBound...].firstIndex(of: ")") else { return nil }
        return UInt32(line[start.upperBound..<end])
    }
    let taskUnit = combinedUnit(task), meshUnit = combinedUnit(mesh)
    #expect(taskUnit != nil)
    #expect(meshUnit != nil)
    #expect(taskUnit != meshUnit)
}
