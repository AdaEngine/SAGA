#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

// These cases exercise the actual object-to-mesh payload ABI, including vec3
// alignment and array stride, rather than only checking reflected byte sizes.
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshHalfPayloadReadback() async throws {
    try await checkMetalNumericPayload(
        feature: .shaderF16, extensionName: "f16",
        members: "head: f16, values: array<vec3<f16>, 2>, tail: f16,",
        writes: "payload.head = 1.5h; payload.values[0] = vec3<f16>(2.0h, 3.0h, 4.0h); payload.values[1] = vec3<f16>(5.0h, 6.0h, 7.0h); payload.tail = 8.0h;",
        reads: "result[0] = u32(payload.head * 2.0h); result[1] = u32(payload.values[0].z); result[2] = u32(payload.values[1].x); result[3] = u32(payload.values[1].z); result[4] = u32(payload.tail);",
        expected: [3, 4, 5, 7, 8])
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshPackedHalfPayloadReadback() async throws {
    try await checkMetalNumericPayload(
        feature: .shaderF16, extensionName: "f16",
        members: "value: vec3<f16>, tail: f16,",
        writes: "payload.value = vec3<f16>(1.0h, 2.0h, 3.0h); payload.tail = 4.0h;",
        reads: "result[0] = u32(payload.value.z); result[1] = u32(payload.value.y); result[2] = u32(payload.value.x); result[3] = u32(payload.tail); result[4] = u32(payload.value.x + payload.tail);",
        expected: [3, 2, 1, 4, 5])
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshInt16PayloadReadback() async throws {
    try await checkMetalNumericPayload(
        feature: .shaderInt16, extensionName: "wgpu_int16",
        members: "head: u16, values: array<vec3<i16>, 2>, tail: u16,",
        writes: "payload.head = u16(65530u); payload.values[0] = vec3<i16>(i16(-11), i16(12), i16(-13)); payload.values[1] = vec3<i16>(i16(21), i16(-22), i16(23)); payload.tail = u16(60000u);",
        reads: "result[0] = u32(payload.head); result[1] = bitcast<u32>(i32(payload.values[0].z)); result[2] = u32(payload.values[1].x); result[3] = u32(payload.values[1].z); result[4] = u32(payload.tail);",
        expected: [65530, UInt32(bitPattern: -13), 21, 23, 60000])
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshInt64PayloadReadback() async throws {
    try await checkMetalNumericPayload(
        feature: .shaderInt64, extensionName: "",
        members: "head: u32, values: array<vec3<u64>, 2>, tail: u64,",
        writes: "payload.head = 17u; payload.values[0] = vec3<u64>(1lu, 2lu, 0x12345678abcdef01lu); payload.values[1] = vec3<u64>(0xfedcba9876543210lu, 4lu, 0x87654321deadbeeflu); payload.tail = 0x13579bdf2468ace0lu;",
        reads: "result[0] = payload.head; result[1] = u32(payload.values[0].z >> 32u); result[2] = u32(payload.values[1].x >> 32u); result[3] = u32(payload.values[1].z >> 32u); result[4] = u32(payload.tail >> 32u);",
        expected: [17, 0x12345678, 0xfedcba98, 0x87654321, 0x13579bdf])
}

private func checkMetalNumericPayload(feature: GPUFeature, extensionName: String, members: String, writes: String, reads: String, expected: [UInt32]) async throws {
    let backend = try MetalBackend.makeDevice()
    guard backend.capabilities.features.contains(.meshShaders), backend.capabilities.features.contains(feature) else { return }
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    \(extensionName.isEmpty ? "" : "enable " + extensionName + ";")
    struct Payload { \(members) }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn task() -> @builtin(mesh_task_size) vec3u { \(writes) return vec3u(1u); }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @group(0) @binding(0) var<storage, read_write> result: array<u32>;
    @mesh(output) @payload(payload) @workgroup_size(1)
    fn mesh() { \(reads) }
    """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(0u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: backend, requiredFeatures: [.meshShaders, feature])
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment]))
        let output = try await device.createBuffer(.init(size: 20, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 20, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), meshBuffers: [.init(binding: 0): .init(buffer: output, size: 20)]),
            .copy(source: output, destination: readback, byteCount: 20),
            .read(buffer: readback, byteCount: 20),
        ]).wait()
        #expect(words == [expected])
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
