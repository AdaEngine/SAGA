import Testing
import SAGATypes

@Test func meshIndirectValidatesArgumentBuffer() throws {
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm)
    let target = ResourceID(device: 1, serial: 2)
    let arguments = ResourceID(device: 1, serial: 3)
    let texture = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment])
    let command = BufferCommand.drawMeshIndirect(pipeline: pipeline, target: target, colorLoad: .load, indirectBuffer: arguments, indirectOffset: 4)
    let good = BufferDescriptor(size: 16, usage: [.indirect])
    try command.validate(device: 1, descriptors: [arguments: good], textures: [target: texture])
    #expect(command.resources.contains(arguments))
    #expect(throws: GPUError.self) { try command.validate(device: 1, descriptors: [arguments: .init(size: 12, usage: [.indirect])], textures: [target: texture]) }
    #expect(throws: GPUError.self) { try command.validate(device: 1, descriptors: [arguments: .init(size: 16, usage: [.storage])], textures: [target: texture]) }
    let misaligned = BufferCommand.drawMeshIndirect(pipeline: pipeline, target: target, colorLoad: .load, indirectBuffer: arguments, indirectOffset: 2)
    #expect(throws: GPUError.self) { try misaligned.validate(device: 1, descriptors: [arguments: good], textures: [target: texture]) }
    let mixed = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1), indirectBuffer: arguments)
    #expect(throws: GPUError.self) { try mixed.validate(device: 1, descriptors: [arguments: good], textures: [target: texture]) }
    let slot = ComputeBindingSlot(binding: 0)
    let writable = MeshPipeline(id: pipeline.id, colorFormat: .rgba8Unorm,
        meshBufferBindings: [.init(slot: slot, minimumByteSize: 4, isReadOnly: false)])
    let aliased = BufferCommand.drawMeshIndirect(pipeline: writable, target: target, colorLoad: .load,
        indirectBuffer: arguments, meshBuffers: [slot: .init(buffer: arguments)])
    #expect(throws: GPUError.self) {
        try aliased.validate(device: 1, descriptors: [arguments: .init(size: 256, usage: [.storage, .indirect])], textures: [target: texture])
    }
}

#if canImport(Metal)
import Metal
import SAGAMetal
import SAGACore
import SAGACompiler
import SAGAShaderIR

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshConsumesGPUAuthoredIndirectArguments() async throws {
    let compiler = IndependentShaderCompiler()
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let producer = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> args: array<u32, 7>;
        @compute @workgroup_size(1) fn main() {
          args[1] = 1u; args[2] = 1u; args[3] = 1u;
          args[4] = 65536u; args[5] = 1u; args[6] = 1u;
        }
        """), to: .msl))
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(1) fn mesh() {
          output.vertex_count = 3u; output.primitive_count = 1u;
          output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
          output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
          output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
          output.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        """)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(1u, 2u, 3u, 255u); }"), entryPoint: "fs", to: .msl)
    let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
    let arguments = try await device.createBuffer(.init(size: 28, usage: [.storage, .indirect]))
    let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let result = try await device.submitCommands([
        .dispatchCompute(pipeline: producer, bindings: [.init(binding: 0): arguments], workgroups: .init(x: 1)),
        .drawMeshIndirect(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), indirectBuffer: arguments, indirectOffset: 16),
        .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .drawMeshIndirect(pipeline: pipeline, target: texture, colorLoad: .load, indirectBuffer: arguments, indirectOffset: 4),
        .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ]).wait()
    #expect(result == [[0], [0xff030201]])
    await device.close()
}
#endif
