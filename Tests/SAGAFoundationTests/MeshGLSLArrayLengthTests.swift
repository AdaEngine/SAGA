import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslMeshAcceptsRuntimeBindingArrayLength() throws {
    let source = """
    enable wgpu_mesh_shader, wgpu_binding_array;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
    var<workgroup> out: O;
    @mesh(out) @workgroup_size(1) fn main() {
      let size = textureDimensions(images[1u]);
      out.vertices[0].position = vec4f(f32(size.x));
      out.vertex_count = 1u; out.primitive_count = 0u;
    }
    """
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileMeshStage(.wgsl(source), entryPoint: "main", to: .glsl,
        bindingArrayLengths: [.init(group: 0, binding: 0): 2])
    guard case .text(_, let glsl) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(glsl.contains("sampler2D"))
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMeshStage(.wgsl(source), entryPoint: "main", to: .glsl,
            bindingArrayLengths: [.init(group: 0, binding: 0): 0])
    }
}
