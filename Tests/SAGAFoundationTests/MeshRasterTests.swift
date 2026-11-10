import SAGACompiler
import SAGAShaderIR

import SAGACore
import SAGATypes
import Testing
@Test func meshRasterRejectsInvalidViewportAndScissor() throws {
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm)
    let target = ResourceID(device: 1, serial: 2)
    let texture = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment])
    for raster in [RenderRasterState(viewport: .init(x: 0, y: 0, width: .nan, height: 1)),
                   RenderRasterState(scissor: .init(x: 3, y: 0, width: 2, height: 1))] {
        let command = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1), raster: raster)
        #expect(throws: GPUError.self) { try command.validate(device: 1, descriptors: [:], textures: [target: texture]) }
    }
}
#if canImport(Metal)
import Metal
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func meshRasterViewportScissorAndCullingReadback() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(1) fn mesh() {
            output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(-1, -1, 0.5, 1);
            output.vertices[1].position = vec4f(3, -1, 0.5, 1);
            output.vertices[2].position = vec4f(-1, 3, 0.5, 1);
            output.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(1u, 2u, 3u, 255u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let front = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint, primitive: .init(frontFace: .counterClockwise, cullMode: .front))
        let back = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint, primitive: .init(frontFace: .clockwise, cullMode: .front))
        let texture = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let buffer = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let raster = RenderRasterState(viewport: .init(x: 0, y: 0, width: 2, height: 1), scissor: .init(x: 0, y: 0, width: 1, height: 1))
        var commands: [BufferCommand] = []
        for candidate in [pipeline, front, back] {
            commands += [.drawMesh(pipeline: candidate, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), raster: raster),
                         .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256), .read(buffer: buffer, byteCount: 8)]
        }
        let results = try await device.submitCommands(commands).wait()
        #expect(results[0] == [0xff030201, 0])
        #expect(Set([results[1][0], results[2][0]]) == Set([0, 0xff030201]))
        #expect(results[1][1] == 0 && results[2][1] == 0)
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
