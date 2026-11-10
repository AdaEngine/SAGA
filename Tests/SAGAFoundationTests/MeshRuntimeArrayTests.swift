import Testing
import SAGACompiler
import SAGAShaderIR

@Test func meshAndTaskReflectRuntimeStorageLengths() throws {
    let resources = """
    struct Data { prefix: vec4u, values: array<u32>, }
    @group(0) @binding(0) var<storage, read> data: Data;
    @group(0) @binding(1) var<storage, read> other: array<u32>;
    fn count() -> u32 { return arrayLength(&data.values) + other[0]; }
    """
    let mesh = """
    enable wgpu_mesh_shader;
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output {
      @builtin(vertices) vertices: array<Vertex, 3>,
      @builtin(primitives) primitives: array<Primitive, 1>,
      @builtin(vertex_count) vertex_count: u32,
      @builtin(primitive_count) primitive_count: u32,
    }
    var<workgroup> output: Output;
    \(resources)
    @mesh(output) @workgroup_size(1) fn main() { output.vertex_count = count(); }
    """
    let task = """
    enable wgpu_mesh_shader;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    \(resources)
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { payload.value = count(); return vec3u(1); }
    """
    let compiler = IndependentShaderCompiler()
    let m = try compiler.compileReflectedMeshStage(.wgsl(mesh), entryPoint: "main", to: .msl)
    let t = try compiler.compileReflectedTaskStage(.wgsl(task), entryPoint: "main", to: .msl)
    for (artifact, buffers) in [(m.artifact, m.storageBindings), (t.artifact, t.storageBindings)] {
        #expect(buffers.map(\.runtimeArrayOffset) == [16, 0])
        #expect(buffers.map(\.runtimeArrayStride) == [4, 4])
        guard case .text(_, let source) = artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains("saga_lengths [[buffer(8)]]"))
        #expect(source.contains("saga_lengths[0]"))
        #expect(source.contains("saga_lengths[1]"))
    }
}

#if canImport(Metal)
import Metal
import SAGAMetal
import SAGACore
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshRuntimeArraysUseEachStageBoundRange() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        @group(0) @binding(0) var<storage, read> input: array<u32>;
        @group(1) @binding(0) var<storage, read_write> result: array<u32>;
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(1)
        fn task() -> @builtin(mesh_task_size) vec3u {
            payload.value = arrayLength(&input) + input[100u]; return vec3u(1u);
        }
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @payload(payload) @workgroup_size(1) fn mesh() {
            result[0] = payload.value * 10u + arrayLength(&result);
            result[100u] = 99u;
        }
        """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(0u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment]))
        let input = try await device.createBuffer(.init(size: 272, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 272, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .write(buffer: input, offset: 256, words: [7, 8, 9, 10]),
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                taskBuffers: [.init(binding: 0): .init(buffer: input, offset: 256, size: 12)],
                meshBuffers: [.init(group: 1, binding: 0): .init(buffer: output, offset: 256, size: 8)]),
            .copy(source: output, sourceOffset: 256, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(result == [[32, 0, 0, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
