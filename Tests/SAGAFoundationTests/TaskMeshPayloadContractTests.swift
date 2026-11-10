import Testing
import SAGABackend
import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes

private actor PayloadContractBackend: MeshBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(
        maxBufferSize: 1024, features: [.meshShaders], maxTaskPayloadSize: 8)
    private(set) var allocations = 0
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws { allocations += 1 }
    func destroyMeshPipeline(id: ResourceID) async {}
}

@Test func taskMeshPayloadLayoutsAndLimitsAreCheckedBeforeAllocation() async throws {
    let compiler = IndependentShaderCompiler()
    func task(_ members: String, preamble: String = "") throws -> TaskStageArtifact {
        try compiler.compileReflectedTaskStage(.wgsl("""
            enable wgpu_mesh_shader;
            \(preamble)
            struct Payload { \(members) }
            var<task_payload> payload: Payload;
            @task @payload(payload) @workgroup_size(1)
            fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1); }
            """), entryPoint: "main", to: .msl)
    }
    func mesh(_ members: String, preamble: String = "") throws -> MeshStageArtifact {
        try compiler.compileReflectedMeshStage(.wgsl("""
            enable wgpu_mesh_shader;
            \(preamble)
            struct Payload { \(members) }
            struct V { @builtin(position) position: vec4f, }
            struct P { @builtin(triangle_indices) indices: vec3u, }
            struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vc: u32, @builtin(primitive_count) pc: u32, }
            var<task_payload> payload: Payload;
            var<workgroup> output: O;
            @mesh(output) @payload(payload) @workgroup_size(1) fn main() {}
            """), entryPoint: "main", to: .msl)
    }
    let backend = PayloadContractBackend()
    let device = try await Device.make(backend: backend)
    let fragment = RenderStageArtifact(artifact: .text(format: .msl, source: "test"), stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs", colorOutputKinds: [0: .float])
    let producer = try task("a: u32, b: f32,")
    let matching = try mesh("x: u32, y: f32,")
    let sameSizeDifferentType = try mesh("x: i32, y: f32,")
    let smaller = try mesh("x: u32,")
    let oversizedTask = try task("a: u32, b: f32, c: u32,")
    let oversizedMesh = try mesh("x: u32, y: f32, z: u32,")
    let nestedTask = try task("value: Inner,", preamble: "struct Inner { a: u32, b: u32, }")
    let nestedMesh = try mesh("value: Inner,", preamble: "struct Inner { x: u32, y: u32, }")
    let arrayMesh = try mesh("value: array<u32, 2>,")
    let vectorMesh = try mesh("value: vec2u,")
    let oversizedArrayMesh = try mesh("value: array<u32, 3>,")
    _ = try await device.createMeshPipeline(task: producer, mesh: matching, fragment: fragment, colorFormat: .rgba8Unorm)
    _ = try await device.createMeshPipeline(task: nestedTask, mesh: nestedMesh, fragment: fragment, colorFormat: .rgba8Unorm)
    for compatible in [arrayMesh, vectorMesh] {
        _ = try await device.createMeshPipeline(task: nestedTask, mesh: compatible, fragment: fragment, colorFormat: .rgba8Unorm)
    }
    _ = try await device.createMeshPipeline(task: producer, mesh: sameSizeDifferentType, fragment: fragment, colorFormat: .rgba8Unorm)
    for invalid in [oversizedArrayMesh] {
        await #expect(throws: GPUError.self) {
            try await device.createMeshPipeline(task: nestedTask, mesh: invalid, fragment: fragment, colorFormat: .rgba8Unorm)
        }
    }
    for invalid in [smaller, oversizedMesh] {
        await #expect(throws: GPUError.self) {
            try await device.createMeshPipeline(task: producer, mesh: invalid, fragment: fragment, colorFormat: .rgba8Unorm)
        }
    }
    await #expect(throws: GPUError.self) {
        try await device.createMeshPipeline(task: oversizedTask, mesh: oversizedMesh, fragment: fragment, colorFormat: .rgba8Unorm)
    }
    await #expect(throws: GPUError.self) {
        try await device.createMeshPipeline(mesh: matching, fragment: fragment, colorFormat: .rgba8Unorm)
    }
    #expect(await backend.allocations == 5)
    await device.close()
}
