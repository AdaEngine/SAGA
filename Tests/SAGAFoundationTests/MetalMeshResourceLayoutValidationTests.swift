#if canImport(Metal)
import Metal
import SAGABackend
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshRejectsReservedBufferIndex() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: O;
        @mesh(output) @workgroup_size(1) fn mesh() { output.vertex_count = 0u; output.primitive_count = 0u; }
        """)
    let compiled = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let mesh = MeshStageArtifact(artifact: compiled.artifact, sourceEntryPoint: compiled.sourceEntryPoint,
        workgroupSize: compiled.workgroupSize, payload: compiled.payload, topology: compiled.topology,
        maxVertices: compiled.maxVertices, maxPrimitives: compiled.maxPrimitives,
        vertexVaryings: compiled.vertexVaryings, primitiveVaryings: compiled.primitiveVaryings,
        storageBindings: [.init(binding: .init(group: 0, binding: 0), flatBindingIndex: 8,
                                minimumByteSize: 4, isReadOnly: true)])
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(1u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    await #expect(throws: GPUError.validation("Invalid Metal object or mesh resource layout")) {
        _ = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
    }
    await device.close()
}
#endif
