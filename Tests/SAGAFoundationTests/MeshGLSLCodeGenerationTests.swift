import Testing
import SAGABackends
import SAGACompiler
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation

@Test func glslMeshWritesOutputAndIndices() throws {
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
    guard case .text(_, let glsl) = try MeshGLSLCodeGenerator().generate(validated, entryPoint: "main") else { Issue.record("Expected GLSL"); return }
    #expect(glsl.contains("#extension GL_EXT_mesh_shader : require"))
    #expect(glsl.contains("SetMeshOutputsEXT(saga_vertex_count, saga_primitive_count)"))
    #expect(glsl.contains("gl_PrimitiveTriangleIndicesEXT"))
    #expect(glsl.contains("gl_MeshVerticesEXT"))
    #expect(glsl.contains("barrier()"))
}

@Test func glslMeshWritesCullPrimitive() throws {
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
    guard case .text(_, let glsl) = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "main", to: .glsl) else { Issue.record("Expected GLSL"); return }
    #expect(glsl.contains("gl_MeshPrimitivesEXT[saga_primitive_index].gl_CullPrimitiveEXT ="))
}

