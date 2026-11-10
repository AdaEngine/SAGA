import SAGAFrontends
import SAGAShaderIR
import SAGAValidation
import Testing

private func meshCullSource(vertex: String = "", primitive: String) -> String { """
  enable wgpu_mesh_shader;
  struct Vertex { @builtin(position) position: vec4f, \(vertex) }
  struct Primitive { @builtin(triangle_indices) indices: vec3u, \(primitive) }
  struct Output {
    @builtin(vertices) vertices: array<Vertex, 3>,
    @builtin(primitives) primitives: array<Primitive, 1>,
  }
  var<workgroup> output: Output;
  @mesh(output) @workgroup_size(1) fn main() {}
  """ }

@Test func meshCullPrimitiveAcceptsBoolOutput() throws {
  let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(
    .wgsl(meshCullSource(primitive: "@builtin(cull_primitive) culled: bool,")))
  try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module)
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(module) }
}

@Test(arguments: [
  "@builtin(cull_primitive) culled: u32,",
  "@builtin(cull_primitive) a: bool, @builtin(cull_primitive) b: bool,",
]) func meshCullPrimitiveRejectsInvalidOutput(primitive: String) throws {
  let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(
    .wgsl(meshCullSource(primitive: primitive)))
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module) }
}

@Test func meshCullPrimitiveRejectsVertexOutput() throws {
  let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(
    .wgsl(meshCullSource(vertex: "@builtin(cull_primitive) culled: bool,", primitive: "")))
  #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator(capabilities: [.meshShader]).validate(module) }
}
