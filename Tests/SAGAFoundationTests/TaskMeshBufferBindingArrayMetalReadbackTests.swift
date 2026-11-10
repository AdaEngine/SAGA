#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskMeshBufferBindingArraysReadback() async throws {
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader, wgpu_binding_array;
    struct Data { values: array<u32>, }
    @group(0) @binding(0) var<storage, read> inputs: binding_array<Data, 2>;
    @group(0) @binding(1) var<storage, read_write> result: array<u32>;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u {
      payload.value = inputs[1].values[0] + arrayLength(&inputs[1].values);
      return vec3u(1u);
    }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: O;
    @mesh(output) @payload(payload) @workgroup_size(1) fn mesh() {
      result[0] = payload.value + inputs[0].values[0] + arrayLength(&inputs[0].values);
    }
    """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(0u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: .renderAttachment))
        let input = try await device.createBuffer(.init(size: 268, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let inputs: [ComputeBindingSlot: StorageBufferBinding] = [
            .init(binding: 0, arrayElement: 0): .init(buffer: input, size: 8),
            .init(binding: 0, arrayElement: 1): .init(buffer: input, offset: 256, size: 12),
        ]
        let meshBindings = inputs.merging([.init(binding: 1): .init(buffer: output, size: 4)]) { _, new in new }
        let missing = inputs.filter { $0.key != .init(binding: 0, arrayElement: 1) }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([
                .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), taskBuffers: missing, meshBuffers: meshBindings),
            ])
        }
        let words = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [7, 8]),
            .write(buffer: input, offset: 256, words: [20, 21, 22]),
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), taskBuffers: inputs, meshBuffers: meshBindings),
            .copy(source: output, destination: readback, byteCount: 4),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(words == [[32]])
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
