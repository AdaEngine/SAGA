import SAGAShaderIR
import SAGAValidation
import SAGAFrontends
import Testing

@Suite struct TaskPayloadAccessTests {
  private func validate(_ entry: String) throws {
    let source = """
      enable wgpu_mesh_shader;
      struct Payload { value: u32, }
      var<task_payload> payload: Payload;
      var<task_payload> other: Payload;
      \(entry)
      """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module)
  }

  @Test func taskReadsAndWritesPayloadThroughHelper() throws {
    try validate("""
      fn update() { payload.value = payload.value + 1u; }
      @task @payload(payload) @workgroup_size(1)
      fn main() -> @builtin(mesh_task_size) vec3u { update(); return vec3u(1); }
      """)
  }

  @Test func taskRejectsUndeclaredPayloadThroughHelper() throws {
    #expect(throws: ShaderFailure.self) {
      try validate("""
        fn update() { other.value = 1u; }
        @task @payload(payload) @workgroup_size(1)
        fn main() -> @builtin(mesh_task_size) vec3u { update(); return vec3u(1); }
        """)
    }
  }

  @Test func computeRejectsPayloadThroughHelper() throws {
    #expect(throws: ShaderFailure.self) {
      try validate("""
        fn read() -> u32 { return payload.value; }
        @compute @workgroup_size(1) fn main() { let x = read(); }
        """)
    }
  }
  @Test func meshCannotWritePayloadVectorElement() throws {
    let source = """
      enable wgpu_mesh_shader;
      struct Payload { value: vec3u, }
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
      fn update() { payload.value.x = 1u; }
      @mesh(output) @payload(payload) @workgroup_size(1)
      fn main() { update(); }
      """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module)
    }
  }
}
