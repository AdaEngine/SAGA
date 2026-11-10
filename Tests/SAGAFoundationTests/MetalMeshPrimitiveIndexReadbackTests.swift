import SAGABackend

import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal

// Adapted from naga/tests/in/wgsl/primitive-index-mesh-shader.wgsl at
// a55352911bb26a1b33bc8d5fc03bdebb0a967f5a.
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshExplicitPrimitiveIndexReachesFragment() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        enable primitive_index;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, @builtin(primitive_index) index: u32, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
          if lane == 0u {
            output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
            output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
            output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
            output.primitives[0].indices = vec3u(0u, 1u, 2u);
            output.primitives[0].index = 7u;
          }
        }
        @fragment fn fs(@builtin(primitive_index) index: u32) -> @location(0) vec4u {
          return vec4u(index, 0u, 0u, 255u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    #expect(mesh.writesPrimitiveIndex)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1)),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff000007]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

private actor PrimitiveIndexMeshBackend: MeshBackendDevice {
    nonisolated let capabilities: DeviceCapabilities
    init(supportsPrimitiveIndex: Bool) throws {
        capabilities = try DeviceCapabilities(maxBufferSize: 1024, features: supportsPrimitiveIndex
            ? [.meshShaders, .primitiveIndex] : [.meshShaders])
    }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] { [] }
    func allocateMeshPipeline(_ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact) async throws {}
    func destroyMeshPipeline(id: ResourceID) async {}
}

@Test(arguments: [false, true])
func meshFragmentPrimitiveIndexRequiresDeviceFeature(supported: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let fragment = try compiler.compileRenderStage(.wgsl("enable primitive_index; @fragment fn fs(@builtin(primitive_index) i: u32) -> @location(0) vec4u { return vec4u(i); }"), entryPoint: "fs", to: .msl)
    let mesh = MeshStageArtifact(artifact: .text(format: .msl, source: "test"), sourceEntryPoint: "mesh", workgroupSize: .init(), payload: nil, topology: .triangles, maxVertices: 3, maxPrimitives: 1, vertexVaryings: [], primitiveVaryings: [], writesPrimitiveIndex: true)
    let device = try await Device.make(backend: PrimitiveIndexMeshBackend(supportsPrimitiveIndex: supported))
    if supported {
        _ = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
    } else {
        await #expect(throws: GPUError.self) {
            try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        }
    }
    await device.close()
}

@Test(arguments: [false, true], [false, true])
func meshPrimitiveIndexInterfaceMustMatch(meshWrites: Bool, fragmentReads: Bool) async throws {
    let primitiveField = meshWrites ? "@builtin(primitive_index) index: u32," : ""
    let source = """
        enable wgpu_mesh_shader;
        enable primitive_index;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, \(primitiveField) }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(1) fn mesh() {}
        """
    let fragmentSource = fragmentReads
        ? "enable primitive_index; @fragment fn fs(@builtin(primitive_index) i: u32) -> @location(0) vec4u { return vec4u(i); }"
        : "@fragment fn fs() -> @location(0) vec4u { return vec4u(1u); }"
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(.wgsl(source), entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl(fragmentSource), entryPoint: "fs", to: .msl)
    #expect(mesh.writesPrimitiveIndex == meshWrites)
    let device = try await Device.make(backend: PrimitiveIndexMeshBackend(supportsPrimitiveIndex: true))
    if meshWrites == fragmentReads {
        _ = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
    } else {
        await #expect(throws: GPUError.self) {
            try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        }
    }
    await device.close()
}
