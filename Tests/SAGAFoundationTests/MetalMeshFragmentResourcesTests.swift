import SAGABackend
import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true), arguments: [false, true])
func metalMeshFragmentUniformReadback(withTask: Bool) async throws {
    let taskDeclarations = withTask ? """
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(1)
        fn task() -> @builtin(mesh_task_size) vec3u { payload.value = 1u; return vec3u(1u); }
        """ : ""
    let taskAttribute = withTask ? "@payload(payload)" : ""
    let laneCondition = withTask ? "lane == 0u && payload.value == 1u" : "lane == 0u"
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        \(taskDeclarations)
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) \(taskAttribute) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
          if \(laneCondition) { output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
            output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
            output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
            output.primitives[0].indices = vec3u(0u, 1u, 2u); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let task = try withTask ? compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl) : nil
    let fragment = try compiler.compileRenderStage(.wgsl("@group(0) @binding(0) var<uniform> value: vec4u; @fragment fn fs() -> @location(0) vec4u { return value; }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                          usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let uniform = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let result = try await device.submitCommands([
            .write(buffer: uniform, offset: 0, words: [1, 2, 3, 255]),
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), fragmentBuffers: [.init(group: 0, binding: 0): .init(buffer: uniform)]),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff030201]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}


@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true), arguments: [false, true])
func metalMeshFragmentSampledTextureReadback(withTask: Bool) async throws {
    let taskDeclarations = withTask ? """
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(1)
        fn task() -> @builtin(mesh_task_size) vec3u { payload.value = 1u; return vec3u(1u); }
        """ : ""
    let taskAttribute = withTask ? "@payload(payload)" : ""
    let laneCondition = withTask ? "lane == 0u && payload.value == 1u" : "lane == 0u"
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        \(taskDeclarations)
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) \(taskAttribute) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
          if \(laneCondition) { output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
            output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
            output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
            output.primitives[0].indices = vec3u(0u, 1u, 2u); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let task = try withTask ? compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl) : nil
    let fragment = try compiler.compileRenderStage(.wgsl("@group(0) @binding(0) var image: texture_2d<f32>; @group(0) @binding(1) var state: sampler; @fragment fn fs() -> @location(0) vec4f { return textureSample(image, state, vec2f(0.5, 0.5)); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource, .textureBinding]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let sampled = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination]))
        let sampler = try await device.createSampler()
        let otherDevice = try await Device.make(backend: MetalBackend.makeDevice())
        let foreign = try await otherDevice.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.textureBinding]))
        let slot = ComputeBindingSlot(binding: 0)
        let samplerSlot = ComputeBindingSlot(binding: 1)
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.drawMesh(pipeline: pipeline, target: texture, colorLoad: .load, workgroups: .init(x: 1), fragmentTextures: [slot: texture], fragmentSamplers: [samplerSlot: sampler])])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.drawMesh(pipeline: pipeline, target: texture, colorLoad: .load, workgroups: .init(x: 1), fragmentTextures: [slot: foreign], fragmentSamplers: [samplerSlot: sampler])])
        }
        await otherDevice.close()
        let result = try await device.submitCommands([
            .writeTexture(destination: sampled, data: [1, 2, 3, 255], region: .init(width: 1, height: 1)),
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), fragmentTextures: [slot: sampled], fragmentSamplers: [samplerSlot: sampler]),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff030201]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

#endif
