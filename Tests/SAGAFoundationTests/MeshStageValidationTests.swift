import SAGAShaderIR
import SAGAValidation
import SAGAFrontends
import Testing

@Suite struct MeshStageValidationTests {
  @Test(arguments: [IRStage.task, .mesh])
  func incompleteStagesNeverProduceValidationWitness(stage: IRStage) throws {
    let function = IRFunction(name: "main", blocks: IRTable([IRBlock(statements: [])]), body: try .init(index: 0))
    let module = IRModule(functions: IRTable([function]), entryPoints: [
      IREntryPoint(name: "main", stage: stage, function: try .init(index: 0))
    ])
    do {
      _ = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.meshShader]))
      Issue.record("Incomplete stage produced a validation witness")
    } catch ShaderFailure.diagnostics(let diagnostics) {
      #expect(diagnostics.first?.code == .invalidBinding)
    }
  }

  @Test func constantTaskWithPayloadValidates() throws {
    let source = """
      enable wgpu_mesh_shader;
      struct Payload { dummy: u32, }
      var<task_payload> payload: Payload;
      @task @payload(payload) @workgroup_size(64)
      fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1); }
      """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module)
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(module) }
  }

  @Test(arguments: [
    "@compute @workgroup_size(1) fn main() { let x = payload.dummy; }",
    "fn read_payload() -> u32 { return payload.dummy; }\n@compute @workgroup_size(1) fn main() { let x = read_payload(); }",
  ])
  func payloadAccessIsRejectedEvenThroughHelpers(entry: String) throws {
    let source = """
      enable wgpu_mesh_shader;
      struct Payload { dummy: u32, }
      var<task_payload> payload: Payload;
      \(entry)
      """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module)
    }
  }
}

@Test func meshOutputMetadataAndInterfaceValidate() throws {
  let source = """
    enable wgpu_mesh_shader;
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output {
      @builtin(vertices) vertices: array<Vertex, 3>,
      @builtin(primitives) primitives: array<Primitive, 1>,
      @builtin(vertex_count) vertex_count: u32,
      @builtin(primitive_count) primitive_count: u32,
    }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(64) fn main() {
      output.vertices[0].position = vec4f(0.0, 0.0, 0.0, 1.0);
      output.primitives[0].indices = vec3u(0u, 0u, 0u);
      output.vertex_count = 1u;
      output.primitive_count = 1u;
    }
    """
  let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
  try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module)
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(module) }
}

@Test(arguments: [
  "@builtin(position) position: vec3f,",
  "@builtin(vertex_index) position: u32,",
]) func meshRejectsInvalidVertexInterface(vertex: String) throws {
  let source = """
    enable wgpu_mesh_shader;
    struct Vertex { \(vertex) }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output {
      @builtin(vertices) vertices: array<Vertex, 3>,
      @builtin(primitives) primitives: array<Primitive, 1>,
      @builtin(vertex_count) vertex_count: u32,
      @builtin(primitive_count) primitive_count: u32,
    }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(1) fn main() {}
    """
  let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module) }
}

@Test func meshOutputCountsAreOptionalButCheckedWhenPresent() throws {
  func shader(_ count: String) -> String { """
    enable wgpu_mesh_shader;
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output {
      @builtin(vertices) vertices: array<Vertex, 3>,
      @builtin(primitives) primitives: array<Primitive, 1>,
      \(count)
    }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(1) fn main() {}
    """ }
  let frontend = WGSLFrontend(options: .init(capabilities: [.meshShader]))
  let withoutCounts = try frontend.parse(.wgsl(shader("")))
  try ShaderSemanticValidator(capabilities: [.meshShader]).validate(withoutCounts)
  let invalidCount = try frontend.parse(.wgsl(shader("@builtin(vertex_count) count: f32,")))
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator(capabilities: [.meshShader]).validate(invalidCount) }
}

@Test(arguments: [
  ("local_invocation_index", "u32"),
  ("local_invocation_id", "vec3u"),
  ("global_invocation_id", "vec3u"),
  ("workgroup_id", "vec3u"),
  ("num_workgroups", "vec3u"),
]) func computeLikeBuiltinsAreValidForTaskAndMesh(builtin: String, type: String) throws {
  let source = """
    enable wgpu_mesh_shader;
    struct Payload { dummy: u32, }
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output {
      @builtin(vertices) vertices: array<Vertex, 3>,
      @builtin(primitives) primitives: array<Primitive, 1>,
    }
    var<task_payload> payload: Payload;
    var<workgroup> output: Output;
    @task @payload(payload) @workgroup_size(1)
    fn task_main(@builtin(\(builtin)) input: \(type)) -> @builtin(mesh_task_size) vec3u { return vec3u(1); }
    @mesh(output) @workgroup_size(1) fn mesh_main(@builtin(\(builtin)) input: \(type)) {}
    """
  let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
  try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module)
}

@Test func computeLikeBuiltinRejectedForVertex() throws {
  let source = """
    @vertex fn main(@builtin(local_invocation_index) input: u32) -> @builtin(position) vec4f {
      return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    """
  let module = try WGSLFrontend().parse(.wgsl(source))
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(module) }
}
