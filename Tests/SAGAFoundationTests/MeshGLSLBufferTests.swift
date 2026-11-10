import Testing
import SAGACompiler
import SAGAShaderIR

@Test func glslMeshBuffersKeepBindingsAndIndependentBounds() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Config { index: u32, }
    struct Input { values: array<u32, 4>, }
    struct Result { values: array<u32, 7>, }
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<Vertex, 3>, @builtin(primitives) primitives: array<Primitive, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    @group(2) @binding(3) var<uniform> config: Config;
    @group(1) @binding(4) var<storage, read> input: Input;
    @group(1) @binding(5) var<storage, read_write> result: Result;
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(1) fn main() {
        let i = config.index;
        result.values[i] = input.values[i];
        output.vertex_count = result.values[i];
        output.primitive_count = 1u;
        output.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """
    let artifact = try IndependentShaderCompiler().compileMeshStage(.wgsl(source), entryPoint: "main", to: .glsl)
    guard case .text(let format, let glsl) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(format == .glsl)
    #expect(glsl.contains("layout(std140, binding = \(ComputeShaderArtifact.uniformBindingBase)) uniform"))
    #expect(glsl.contains("layout(std430, binding = 0) readonly buffer"))
    #expect(glsl.contains("layout(std430, binding = 1) buffer"))
    #expect(glsl.contains("< 4u"))
    #expect(glsl.contains("< 7u"))
}
