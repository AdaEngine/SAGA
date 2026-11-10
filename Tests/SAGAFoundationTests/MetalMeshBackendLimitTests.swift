#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalBackendRejectsMeshDispatchBeyondItsLimit() async throws {
    let backend = try MetalBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let mesh = try compiler.compileReflectedMeshStage(.wgsl("""
            enable wgpu_mesh_shader;
            struct V { @builtin(position) position: vec4f, }
            struct P { @builtin(triangle_indices) indices: vec3u, }
            struct Output {
                @builtin(vertices) vertices: array<V, 3>,
                @builtin(primitives) primitives: array<P, 1>,
                @builtin(vertex_count) vertex_count: u32,
                @builtin(primitive_count) primitive_count: u32,
            }
            var<workgroup> output: Output;
            @mesh(output) @workgroup_size(1) fn mesh() {
                output.vertex_count = 0u; output.primitive_count = 0u;
            }
            """), entryPoint: "mesh", to: .msl)
        let fragment = try compiler.compileRenderStage(
            .wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(0u); }"),
            entryPoint: "fs", to: .msl)
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: .renderAttachment))
        let limit = backend.capabilities.maxMeshWorkgroupsPerDimension
        guard limit < UInt32.max else { await device.close(); return }
        // A zero dimension prevents an accidental large GPU dispatch before the fix.
        let command = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load,
                                             workgroups: .init(x: limit + 1, y: 0))
        await #expect(throws: GPUError.self) {
            try await backend.submitCommands(.init(sequence: 1, resources: command.resources), commands: [command])
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
