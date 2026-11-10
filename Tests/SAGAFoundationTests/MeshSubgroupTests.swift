import Testing
import SAGACompiler
import SAGAShaderIR

@Test func meshSubgroupMetalCodeGeneration() throws {
    let source = """
    enable wgpu_mesh_shader;
    enable subgroups;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(32) fn main(@builtin(subgroup_invocation_id) lane: u32) {
      let total = subgroupAdd(lane);
      if lane == 0u { output.vertex_count = 3u; output.primitive_count = 1u; output.vertices[0].position = vec4f(f32(total)); output.primitives[0].indices = vec3u(0u, 1u, 2u); }
    }
    """
    let shader = try IndependentShaderCompiler().compileReflectedMeshStage(.wgsl(source), entryPoint: "main", to: .msl)
    guard case .text(_, let text) = shader.artifact else { Issue.record("Expected Metal source"); return }
    #expect(text.contains("thread_index_in_simdgroup"))
    #expect(text.contains("simd_sum("))
}

import SAGABackend
import SAGACore
import SAGATypes

private actor MeshSubgroupNoFeatureBackend: MeshBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders])
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws { Issue.record("Backend must not receive unsupported subgroup shader") }
    func destroyMeshPipeline(id: ResourceID) async {}
}

@Test func meshSubgroupRequiresDeviceFeature() async throws {
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [], requiresSubgroups: true)
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs", colorOutputKinds: [0: .float])
    let device = try await Device.make(backend: MeshSubgroupNoFeatureBackend())
    await #expect(throws: GPUError.self) { try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm) }
    await device.close()
}

@Test func taskSubgroupRejectsDivergentControl() {
    let source = """
    enable wgpu_mesh_shader;
    enable subgroups;
    struct Payload { value: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(32)
    fn main(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
      if lane == 0u { payload.value = subgroupAdd(lane); }
      return vec3u(1u);
    }
    """
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl) }
}

@Test func taskSubgroupRequiresDeviceFeature() async throws {
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let task = TaskStageArtifact(artifact: code, sourceEntryPoint: "task", workgroupSize: .init(), payload: nil, requiresSubgroups: true)
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [])
    let fragment = RenderStageArtifact(artifact: code, stage: .fragment, sourceEntryPoint: "fs", entryPoint: "fs", colorOutputKinds: [0: .float])
    let device = try await Device.make(backend: MeshSubgroupNoFeatureBackend())
    await #expect(throws: GPUError.self) { try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm) }
    await device.close()
}

@Test func meshSubgroupRejectsDivergentControl() {
    let source = """
    enable wgpu_mesh_shader;
    enable subgroups;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @mesh(output) @workgroup_size(32) fn main(@builtin(local_invocation_index) lane: u32) {
      if lane == 0u { output.vertex_count = subgroupAdd(lane); }
    }
    """
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compileReflectedMeshStage(.wgsl(source), entryPoint: "main", to: .msl) }
}

#if canImport(Metal)
import Metal
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskAndMeshSubgroupsReadBack() async throws {
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    enable subgroups;
    struct Payload { value: u32, size: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(32)
    fn task(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32, @builtin(local_invocation_index) index: u32) -> @builtin(mesh_task_size) vec3u {
      let sum = subgroupAdd(lane);
      if index == 0u { payload.value = sum; payload.size = size; }
      return vec3u(1u);
    }
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: Output;
    @group(0) @binding(0) var<storage, read_write> result: array<u32>;
    @mesh(output) @payload(payload) @workgroup_size(32)
    fn mesh(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32, @builtin(local_invocation_index) index: u32) {
      let sum = subgroupAdd(lane);
      if index == 0u { result[0] = payload.value; result[1] = payload.size; result[2] = sum; result[3] = size; }
    }
    """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    #expect(task.requiresSubgroups && mesh.requiresSubgroups)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(0u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
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
        #expect(words.count == 1 && words[0].count == 4)
        if words.count == 1 && words[0].count == 4 {
            let values = words[0]
            #expect(values[1] > 0 && values[3] > 0)
            if values[1] > 0 && values[3] > 0 {
                let taskActive = min(values[1], 32), meshActive = min(values[3], 32)
                #expect(values[0] == taskActive * (taskActive - 1) / 2)
                #expect(values[2] == meshActive * (meshActive - 1) / 2)
            }
        }
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
