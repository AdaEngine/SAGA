#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshFragmentAcceptsSampledTextureBindingArray() async throws {
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(.wgsl("""
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> out: O;
    @mesh(out) @workgroup_size(1) fn main() {
      out.vertex_count = 0u; out.primitive_count = 0u;
    }
    """), entryPoint: "main", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("""
    enable wgpu_binding_array;
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
    @fragment fn fs() -> @location(0) vec4f {
      return textureLoad(images[0], vec2i(0), 0) + textureLoad(images[1], vec2i(0), 0);
    }
    """), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        _ = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
