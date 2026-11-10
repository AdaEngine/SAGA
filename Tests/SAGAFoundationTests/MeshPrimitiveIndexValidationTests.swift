import SAGAFrontends
import SAGAShaderIR
import SAGAValidation
import Testing

private func meshPrimitiveIndexSource(vertex: String = "", primitive: String) -> String { """
  enable wgpu_mesh_shader;
  enable primitive_index;
  struct Vertex { @builtin(position) position: vec4f, \(vertex) }
  struct Primitive { @builtin(triangle_indices) indices: vec3u, \(primitive) }
  struct Output {
    @builtin(vertices) vertices: array<Vertex, 3>,
    @builtin(primitives) primitives: array<Primitive, 1>,
  }
  var<workgroup> output: Output;
  @mesh(output) @workgroup_size(1) fn main() {}
  """ }

@Test func meshPrimitiveIndexAcceptsPrimitiveOutput() throws {
  let caps: ShaderCapabilities = [.meshShader, .primitiveIndex]
  let module = try WGSLFrontend(options: .init(capabilities: caps)).parse(
    .wgsl(meshPrimitiveIndexSource(primitive: "@builtin(primitive_index) index: u32,")))
  try ShaderSemanticValidator(capabilities: caps).validate(module)
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module) }
}

// Adapted from naga/tests/in/wgsl/primitive-index-mesh-shader.wgsl at
// a55352911bb26a1b33bc8d5fc03bdebb0a967f5a; exercises both entry points.
@Test func meshPrimitiveIndexAcceptsNagaCorpusShader() throws {
  let source = """
    enable wgpu_mesh_shader;
    enable primitive_index;
    struct VertexOutput { @builtin(position) position: vec4<f32>, }
    struct PrimitiveOutput {
      @builtin(triangle_indices) indices: vec3<u32>,
      @builtin(primitive_index) index: u32,
    }
    struct MeshOutput {
      @builtin(vertices) vertices: array<VertexOutput, 3>,
      @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
      @builtin(vertex_count) vertex_count: u32,
      @builtin(primitive_count) primitive_count: u32,
    }
    var<workgroup> mesh_output: MeshOutput;
    @mesh(mesh_output) @workgroup_size(1) fn ms_main() {
      mesh_output.vertex_count = 3;
      mesh_output.primitive_count = 1;
      mesh_output.vertices[0].position = vec4(0., 1., 0., 1.);
      mesh_output.vertices[1].position = vec4(-1., -1., 0., 1.);
      mesh_output.vertices[2].position = vec4(1., -1., 0., 1.);
      mesh_output.primitives[0].indices = vec3<u32>(0, 1, 2);
      mesh_output.primitives[0].index = 7u;
    }
    @fragment fn fs_main(@builtin(primitive_index) index: u32) -> @location(0) vec4<f32> {
      return vec4(f32(index), 1., 1., 1.);
    }
    """
  let caps: ShaderCapabilities = [.meshShader, .primitiveIndex]
  let module = try WGSLFrontend(options: .init(capabilities: caps)).parse(.wgsl(source))
  try ShaderSemanticValidator(capabilities: caps).validate(module)
}

@Test(arguments: [
  "@builtin(primitive_index) index: i32,",
  "@builtin(primitive_index) a: u32, @builtin(primitive_index) b: u32,",
]) func meshPrimitiveIndexRejectsInvalidPrimitiveOutput(primitive: String) throws {
  let caps: ShaderCapabilities = [.meshShader, .primitiveIndex]
  let module = try WGSLFrontend(options: .init(capabilities: caps)).parse(.wgsl(meshPrimitiveIndexSource(primitive: primitive)))
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator(capabilities: caps).validate(module) }
}

@Test func meshPrimitiveIndexRejectsVertexOutput() throws {
  let caps: ShaderCapabilities = [.meshShader, .primitiveIndex]
  let module = try WGSLFrontend(options: .init(capabilities: caps)).parse(
    .wgsl(meshPrimitiveIndexSource(vertex: "@builtin(primitive_index) index: u32,", primitive: "")))
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator(capabilities: caps).validate(module) }
}
