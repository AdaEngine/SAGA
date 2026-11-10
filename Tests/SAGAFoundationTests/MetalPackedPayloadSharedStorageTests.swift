#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalPackedPayloadSharedStorageAggregateCopies() async throws {
    let backend = try MetalBackend.makeDevice()
    guard backend.capabilities.features.contains(.meshShaders) else { return }
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    struct Item { value: vec3u, tail: u32, }
    struct Payload { item: Item, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u {
        let local = Item(vec3u(11u, 12u, 13u), 14u);
        payload.item = local;
        return vec3u(1u);
    }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @group(0) @binding(0) var<storage, read_write> items: array<Item>;
    @mesh(output) @payload(payload) @workgroup_size(1)
    fn mesh() {
        let from_storage = items[0];
        let from_payload = payload.item;
        items[0] = from_payload;
        items[1] = from_storage;
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
        let storage = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .write(buffer: storage, offset: 0, words: [21, 22, 23, 24, 0, 0, 0, 0]),
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), meshBuffers: [.init(binding: 0): .init(buffer: storage, size: 32)]),
            .copy(source: storage, destination: readback, byteCount: 32),
            .read(buffer: readback, byteCount: 32),
        ]).wait()
        #expect(words == [[11, 12, 13, 14, 21, 22, 23, 24]])
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
