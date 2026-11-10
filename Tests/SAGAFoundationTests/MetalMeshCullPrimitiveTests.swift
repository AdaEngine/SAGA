import SAGABackend
import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshCullPrimitiveControlsRasterization() async throws {
    let compiler = IndependentShaderCompiler()
    func mesh(_ culled: Bool) throws -> MeshStageArtifact {
        try compiler.compileReflectedMeshStage(.wgsl("""
            enable wgpu_mesh_shader;
            struct V { @builtin(position) position: vec4f, }
            struct P { @builtin(triangle_indices) indices: vec3u, @builtin(cull_primitive) culled: bool, }
            struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
            var<workgroup> output: Output;
            @mesh(output) @workgroup_size(1) fn mesh() {
                output.vertex_count = 3u; output.primitive_count = 1u;
                output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
                output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
                output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
                output.primitives[0].indices = vec3u(0u, 1u, 2u);
                output.primitives[0].culled = \(culled ? "true" : "false");
            }
            """), entryPoint: "mesh", to: .msl)
    }
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(1u, 2u, 3u, 255u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        for (culled, expected) in [(false, UInt32(0xff030201)), (true, UInt32(0))] {
            let pipeline = try await device.createMeshPipeline(mesh: mesh(culled), fragment: fragment, colorFormat: .rgba8Uint)
            let result = try await device.submitCommands([
                .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1)),
                .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
                .read(buffer: readback, byteCount: 4),
            ]).wait()
            #expect(result == [[expected]])
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
