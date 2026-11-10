import Testing
import SAGACompiler
import SAGAShaderIR

@Test func meshMetalStorageTextureReflection() throws {
    func shader(format: String, value: String, access: String = "write") -> String { """
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> out: O;
    @group(0) @binding(3) var image: texture_storage_2d<\(format), \(access)>;
    @mesh(out) @workgroup_size(1) fn main() {
      textureStore(image, vec2i(0), \(value));
      out.vertex_count = 3u; out.primitive_count = 1u; out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """ }
    let result = try IndependentShaderCompiler().compileReflectedMeshStage(.wgsl(shader(format: "rgba8unorm", value: "vec4f(1.0)")), entryPoint: "main", to: .msl)
    #expect(result.storageTextures.count == 1)
    guard case .text(_, let text) = result.artifact else { Issue.record("Expected Metal source"); return }
    let index = try #require(result.storageTextures.first?.flatBindingIndex)
    #expect(text.contains("[[texture(\(index))]]"))
    let integer = try IndependentShaderCompiler().compileReflectedMeshStage(.wgsl(shader(format: "rgba8uint", value: "vec4u(1u)")), entryPoint: "main", to: .msl)
    #expect(integer.storageTextures.first?.format == .rgba8Uint)
    let readWrite = shader(format: "r32uint", value: "textureLoad(image, vec2i(0)) + vec4u(1u)", access: "read_write")
    let readable = try IndependentShaderCompiler().compileReflectedMeshStage(.wgsl(readWrite), entryPoint: "main", to: .msl)
    #expect(readable.storageTextures.first?.access == [.load, .store])
    #expect(readable.storageTextures.requiredStorageTextureFeatures.contains(.storageTextureReadWrite))
}

#if canImport(Metal)
import Metal
import SAGAMetal
import SAGACore
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshStorageTextureReadback() async throws {
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> out: O;
    @group(0) @binding(3) var image: texture_storage_2d<rgba8unorm, write>;
    @mesh(out) @workgroup_size(1) fn main() {
      textureStore(image, vec2i(0), vec4f(1.0, 0.0, 0.0, 1.0));
      out.vertex_count = 3u; out.primitive_count = 1u; out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "main", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4f { return vec4f(0.0); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.storageBinding, .copySource]))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), meshTextures: [.init(binding: 3): image]),
            .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(try #require(result.first?.first) == 0xff0000ff)
    } catch { await device.close(); throw error }
    await device.close()
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshIntegerStorageTextureReadback() async throws {
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> out: O;
    @group(0) @binding(3) var image: texture_storage_2d<rgba8uint, write>;
    @mesh(out) @workgroup_size(1) fn main() {
      textureStore(image, vec2i(0), vec4u(7u, 13u, 29u, 255u));
      out.vertex_count = 3u; out.primitive_count = 1u; out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "main", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4f { return vec4f(0.0); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.storageBinding, .copySource]))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), meshTextures: [.init(binding: 3): image]),
            .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(try #require(result.first?.first) == 0xff1d0d07)
    } catch { await device.close(); throw error }
    await device.close()
}
#endif

import SAGATypes
import SAGABackend

@Test func meshStorageTextureValidatesOwnershipUsageAndAliases() throws {
    let slot = ComputeBindingSlot(binding: 3)
    let requirement = StorageTextureRequirement(slot: slot, format: .rgba8Unorm, access: .writeOnly)
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm, meshStorageTextures: [requirement])
    let target = ResourceID(device: 1, serial: 2), image = ResourceID(device: 1, serial: 3)
    let wrong = ResourceID(device: 2, serial: 3)
    let textures = [target: TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .storageBinding]),
                    image: TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .storageBinding)]
    func draw(_ id: ResourceID) -> BufferCommand {
        .drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1), meshTextures: [slot: id])
    }
    #expect(draw(image).resources.contains(image))
    try draw(image).validate(device: 1, descriptors: [:], textures: textures)
    #expect(throws: GPUError.self) { try draw(target).validate(device: 1, descriptors: [:], textures: textures) }
    #expect(throws: GPUError.self) { try draw(wrong).validate(device: 1, descriptors: [:], textures: textures) }
    #expect(throws: GPUError.self) { try draw(image).validate(device: 1, descriptors: [:], textures: [target: textures[target]!]) }
}
