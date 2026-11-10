#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskAndMeshHalfArithmeticReadBack() async throws {
    let backend = try MetalBackend.makeDevice()
    guard backend.capabilities.features.contains(.meshShaders),
          backend.capabilities.features.contains(.shaderF16) else { return }

    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader, f16;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn task(@builtin(local_invocation_index) index: u32) -> @builtin(mesh_task_size) vec3u {
      let value = f16(index + 2u) * 1.5h;
      payload.value = u32(value);
      return vec3u(1u);
    }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @group(0) @binding(0) var<storage, read_write> result: array<u32>;
    @mesh(output) @payload(payload) @workgroup_size(1)
    fn mesh() {
      let value = f16(payload.value) + 2.0h;
      result[0] = payload.value;
      result[1] = u32(value);
    }
    """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(0u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: backend, requiredFeatures: [.meshShaders, .shaderF16])
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment]))
        let output = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), meshBuffers: [.init(binding: 0): .init(buffer: output, size: 8)]),
            .copy(source: output, destination: readback, byteCount: 8),
            .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(words == [[3, 5]])
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
