#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalPackedPayloadVectorOperationsAndSharedStructureReadback() async throws {
    let backend = try MetalBackend.makeDevice()
    guard backend.capabilities.features.contains(.meshShaders) else { return }
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    struct Item { value: vec3u, tail: u32, }
    struct Payload { item: Item, }
    var<task_payload> payload: Payload;
    fn update() {
        let rotated = payload.item.value.zyx;
        payload.item.value = rotated + vec3u(10u, 20u, 30u);
        payload.item.value.y = payload.item.value.y + 1u;
        payload.item.tail = payload.item.tail + 4u;
    }
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u {
        var local_item = Item(vec3u(1u, 2u, 3u), 40u);
        payload.item = local_item;
        update();
        return vec3u(1u);
    }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @group(0) @binding(0) var<storage, read_write> result: array<u32>;
    @mesh(output) @payload(payload) @workgroup_size(1)
    fn mesh() {
        let reversed = payload.item.value.zyx;
        result[0] = reversed.x; result[1] = reversed.y; result[2] = reversed.z;
        result[3] = payload.item.tail;
    }
    """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(0u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: backend, requiredFeatures: [.meshShaders])
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), meshBuffers: [.init(binding: 0): .init(buffer: output, size: 16)]),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(words == [[31, 23, 13, 44]])
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
