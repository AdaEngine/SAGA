import Testing
import SAGABackend
import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes

private actor VaryingBackend: MeshBackendDevice {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders])
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyMeshPipeline(id: ResourceID) async {}
}

@Test func meshVaryingLinksOnlyMatchingFragmentInputs() async throws {
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1,
        vertexVaryings: [.init(location: 0, scalar: .f32, componentCount: 4)], primitiveVaryings: [])
    func fragment(_ source: String) throws -> RenderStageArtifact {
        try IndependentShaderCompiler().compileRenderStage(.wgsl(source), entryPoint: "fs", to: .msl)
    }
    let device = try await Device.make(backend: VaryingBackend())
    let good = try fragment("@fragment fn fs(@location(0) value: vec4f) -> @location(0) vec4f { return value; }")
    _ = try await device.createMeshPipeline(mesh: mesh, fragment: good, colorFormat: .rgba8Unorm)
    for source in [
        "@fragment fn fs(@location(1) value: vec4f) -> @location(0) vec4f { return value; }",
        "@fragment fn fs(@location(0) value: vec3f) -> @location(0) vec4f { return vec4f(value, 1.0); }",
        "@fragment fn fs(@location(0) @interpolate(flat) value: vec4f) -> @location(0) vec4f { return value; }",
    ] {
        let bad = try fragment(source)
        await #expect(throws: GPUError.self) { try await device.createMeshPipeline(mesh: mesh, fragment: bad, colorFormat: .rgba8Unorm) }
    }
    let primitive = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [],
        primitiveVaryings: [.init(location: 0, scalar: .f32, componentCount: 4)])
    let flat = try fragment("@fragment fn fs(@location(0) @interpolate(flat) value: vec4f) -> @location(0) vec4f { return value; }")
    _ = try await device.createMeshPipeline(mesh: primitive, fragment: flat, colorFormat: .rgba8Unorm)
    await #expect(throws: GPUError.self) {
        try await device.createMeshPipeline(mesh: primitive, fragment: good, colorFormat: .rgba8Unorm)
    }
    let marked = try fragment("enable wgpu_mesh_shader; @fragment fn fs(@location(0) @per_primitive value: vec4f) -> @location(0) vec4f { return value; }")
    _ = try await device.createMeshPipeline(mesh: primitive, fragment: marked, colorFormat: .rgba8Unorm)
    await #expect(throws: GPUError.self) {
        try await device.createMeshPipeline(mesh: mesh, fragment: marked, colorFormat: .rgba8Unorm)
    }
    let duplicate = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1,
        vertexVaryings: [.init(location: 0, scalar: .f32, componentCount: 4)],
        primitiveVaryings: [.init(location: 0, scalar: .f32, componentCount: 4)])
    await #expect(throws: GPUError.self) {
        try await device.createMeshPipeline(mesh: duplicate, fragment: flat, colorFormat: .rgba8Unorm)
    }
    await device.close()
}

@Test func meshVaryingInterpolationMustMatch() async throws {
    let code = ShaderArtifact.text(format: .msl, source: "test")
    let mesh = MeshStageArtifact(artifact: code, sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil,
        topology: .triangles, maxVertices: 3, maxPrimitives: 1,
        vertexVaryings: [.init(location: 0, scalar: .f32, componentCount: 2, interpolation: .linear, sampling: .centroid)], primitiveVaryings: [])
    let compiler = IndependentShaderCompiler()
    let good = try compiler.compileRenderStage(.wgsl("@fragment fn fs(@location(0) @interpolate(linear, centroid) v: vec2f) -> @location(0) vec4f { return vec4f(v, 0.0, 1.0); }"), entryPoint: "fs", to: .msl)
    let bad = try compiler.compileRenderStage(.wgsl("@fragment fn fs(@location(0) v: vec2f) -> @location(0) vec4f { return vec4f(v, 0.0, 1.0); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: VaryingBackend())
    _ = try await device.createMeshPipeline(mesh: mesh, fragment: good, colorFormat: .rgba8Unorm)
    await #expect(throws: GPUError.self) { try await device.createMeshPipeline(mesh: mesh, fragment: bad, colorFormat: .rgba8Unorm) }
    await device.close()
}
