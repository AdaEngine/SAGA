#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let fragmentArrayMeshSource = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: O;
    @mesh(output) @workgroup_size(1) fn mesh() {
      output.vertex_count = 3u; output.primitive_count = 1u;
      output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
      output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
      output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
      output.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """)

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshFragmentTextureSamplerArraysReadback() async throws {
    let fragmentSource = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
        @group(0) @binding(1) var states: binding_array<sampler, 2>;
        @fragment fn fs() -> @location(0) vec4f {
          let a = textureSample(images[0], states[0], vec2f(1.25, 0.5)).r;
          let b = textureSample(images[1], states[1], vec2f(1.25, 0.5)).r;
          return vec4f(a, b, 0.0, 1.0);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(fragmentArrayMeshSource, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(fragmentSource, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let input = TextureDescriptor(width: 2, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination])
        let a = try await device.createTexture(input)
        let b = try await device.createTexture(input)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let clamp = try await device.createSampler(.init())
        let repeating = try await device.createSampler(.init(addressModeU: .repeat))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let textures: [ComputeBindingSlot: ResourceID] = [.init(binding: 0, arrayElement: 0): a, .init(binding: 0, arrayElement: 1): b]
        let samplers: [ComputeBindingSlot: ResourceID] = [.init(binding: 1, arrayElement: 0): clamp, .init(binding: 1, arrayElement: 1): repeating]
        let missing = textures.filter { $0.key != .init(binding: 0, arrayElement: 1) }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1), fragmentTextures: missing, fragmentSamplers: samplers)])
        }
        let region = TextureCopyRegion(width: 2, height: 1)
        let words = try await device.submitCommands([
            .writeTexture(destination: a, data: [10, 0, 0, 255, 70, 0, 0, 255], region: region),
            .writeTexture(destination: b, data: [150, 0, 0, 255, 220, 0, 0, 255], region: region),
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), fragmentTextures: textures, fragmentSamplers: samplers),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(words == [[0xff009646]])
    } catch { await device.close(); throw error }
    await device.close()
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshFragmentBufferArraysReadback() async throws {
    let fragmentSource = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct Data { values: array<u32>, }
        @group(0) @binding(0) var<storage, read> inputs: binding_array<Data, 2>;
        @fragment fn fs() -> @location(0) vec4u {
          return vec4u(inputs[0].values[0], inputs[1].values[0], arrayLength(&inputs[0].values), arrayLength(&inputs[1].values));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(fragmentArrayMeshSource, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(fragmentSource, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let input = try await device.createBuffer(.init(size: 268, usage: [.storage, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let buffers: [ComputeBindingSlot: StorageBufferBinding] = [
            .init(binding: 0, arrayElement: 0): .init(buffer: input, size: 8),
            .init(binding: 0, arrayElement: 1): .init(buffer: input, offset: 256, size: 12),
        ]
        let missing = buffers.filter { $0.key != .init(binding: 0, arrayElement: 1) }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1), fragmentBuffers: missing)])
        }
        let words = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [7, 8]),
            .write(buffer: input, offset: 256, words: [20, 21, 22]),
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), fragmentBuffers: buffers),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(words == [[0x03021407]])
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
