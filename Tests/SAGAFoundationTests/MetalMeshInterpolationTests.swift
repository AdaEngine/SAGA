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
func metalMeshLinearCentroidVaryingReadback() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, @location(0) @interpolate(linear, centroid) value: f32, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
          if lane == 0u { output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
            output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
            output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
            output.vertices[0].value = 0.0; output.vertices[1].value = 0.4; output.vertices[2].value = 0.8;
            output.primitives[0].indices = vec3u(0u, 1u, 2u); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    #expect(mesh.vertexVaryings.first?.interpolation == .linear)
    #expect(mesh.vertexVaryings.first?.sampling == .centroid)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs(@location(0) @interpolate(linear, centroid) value: f32) -> @location(0) vec4f { return vec4f(value, 0.0, 0.0, 1.0); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1)),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        let red = try #require(result.first?.first).littleEndian & 0xff
        #expect((76...77).contains(red))
        #expect(((result.first?.first ?? 0) & 0xff000000) == 0xff000000)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
