import SAGACompiler
import SAGAShaderIR
import Testing

@Test func hlslMeshVertexInterpolationQualifiers() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V {
          @builtin(position) position: vec4f,
          @location(0) @interpolate(perspective, center) a: f32,
          @location(1) @interpolate(perspective, centroid) b: f32,
          @location(2) @interpolate(perspective, sample) c: f32,
          @location(3) @interpolate(linear, center) d: f32,
          @location(4) @interpolate(linear, centroid) e: f32,
          @location(5) @interpolate(linear, sample) f: f32,
          @location(6) @interpolate(flat) g: u32,
        }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(1) fn mesh() {}
        """)
    let mesh = try IndependentShaderCompiler().compileMeshStage(source, entryPoint: "mesh", to: .hlsl)
    guard case .text(.hlsl, let code) = mesh else { throw ShaderFailure.invalidHandle }
    #expect(code.contains("float saga_field_1 : TEXCOORD0;"))
    #expect(code.contains("centroid float saga_field_2 : TEXCOORD1;"))
    #expect(code.contains("sample float saga_field_3 : TEXCOORD2;"))
    #expect(code.contains("noperspective float saga_field_4 : TEXCOORD3;"))
    #expect(code.contains("noperspective centroid float saga_field_5 : TEXCOORD4;"))
    #expect(code.contains("noperspective sample float saga_field_6 : TEXCOORD5;"))
    #expect(code.contains("nointerpolation uint saga_field_7 : TEXCOORD6;"))
}
