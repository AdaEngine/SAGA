import SAGAFrontends
import SAGAShaderIR
import Testing

// Adapted from Naga's mesh-shader-empty.wgsl at the pinned revision.
private let meshSource = """
enable wgpu_mesh_shader;
struct Payload { dummy: u32, }
struct Vertex { @builtin(position) position: vec4f, }
struct Primitive { @builtin(triangle_indices) indices: vec3u, }
struct Output {
  @builtin(vertices) vertices: array<Vertex, 3>,
  @builtin(primitives) primitives: array<Primitive, 1>,
  @builtin(vertex_count) vertex_count: u32,
  @builtin(primitive_count) primitive_count: u32,
}
var<task_payload> payload: Payload;
var<workgroup> output: Output;
@task @payload(payload) @workgroup_size(64)
fn task_main() -> @builtin(mesh_task_size) vec3u { return vec3u(1); }
@mesh(output) @payload(payload) @workgroup_size(64) fn mesh_main() {}
"""

@Test func wgslMeshFrontendPreservesEntryMetadata() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(meshSource))
    let task = try #require(module.entryPoints.first { $0.stage == .task })
    let mesh = try #require(module.entryPoints.first { $0.stage == .mesh })
    #expect(task.taskPayload == mesh.taskPayload)
    #expect(task.taskPayload != nil)
    #expect(mesh.meshInfo?.maxVertices == 3)
    #expect(mesh.meshInfo?.maxPrimitives == 1)
    #expect(mesh.meshInfo?.topology == .triangles)
    #expect(mesh.workgroupSize.x == 64)
}

@Test func wgslMeshFrontendRequiresCapabilityAndExtension() throws {
    #expect(throws: ShaderFailure.self) {
        try WGSLFrontend(options: .init(capabilities: [])).parse(.wgsl(meshSource))
    }
    #expect(throws: ShaderFailure.self) {
        try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl("@task @workgroup_size(1) fn t() {}"))
    }
}
