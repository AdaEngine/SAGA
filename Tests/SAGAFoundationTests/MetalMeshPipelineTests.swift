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
func metalMeshPipelineDrawsAndReadsBack(withTask: Bool) async throws {
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
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(1u, 2u, 3u, 255u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                          usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1)),
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

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshMultipleTargetsReadback() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
          if lane == 0u { output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
            output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
            output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
            output.primitives[0].indices = vec3u(0u, 1u, 2u); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("struct F { @location(0) a: vec4u, @location(1) b: vec4u, } @fragment fn fs() -> F { return F(vec4u(1u, 2u, 3u, 255u), vec4u(4u, 5u, 6u, 255u)); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormats: [0: .rgba8Uint, 1: .rgba8Uint])
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                          usage: [.renderAttachment, .copySource]))
        let second = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let secondReadback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), additionalColors: [1: .init(texture: second, loadAction: .clear(.transparentBlack))]),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: second, destination: secondReadback, bytesPerRow: 256),
            .read(buffer: secondReadback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff030201], [0xff060504]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

#if canImport(Metal)
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshInterpolatesVaryingIntoFragmentReadback() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, @location(0) value: f32, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
          if lane == 0u { output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
            output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
            output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
            output.vertices[0].value = 0.0; output.vertices[1].value = 0.4; output.vertices[2].value = 0.8;
            output.primitives[0].indices = vec3u(0u, 1u, 2u); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs(@location(0) value: f32) -> @location(0) vec4f { return vec4f(value, 0.0, 0.0, 1.0); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1)),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        let red = try #require(result.first?.first).littleEndian & 0xff
        #expect((76...77).contains(red))
        #expect(((result.first?.first ?? 0) & 0xff000000) == 0xff000000)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

#if canImport(Metal)
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskAndMeshBuffersReachGPU() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Data { value: u32, }
        struct Payload { value: u32, }
        @group(0) @binding(0) var<uniform> input: Data;
        @group(1) @binding(0) var<storage, read_write> result: Data;
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(1)
        fn task() -> @builtin(mesh_task_size) vec3u { payload.value = input.value; return vec3u(1u); }
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @payload(payload) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
          if lane == 0u { result.value = payload.value + 7u; output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
            output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
            output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
            output.primitives[0].indices = vec3u(0u, 1u, 2u); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(1u, 2u, 3u, 255u); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: [.renderAttachment, .copySource]))
        let input = try await device.createBuffer(.init(size: 260, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 260, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .write(buffer: input, offset: 256, words: [35]),
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                      taskBuffers: [.init(binding: 0): .init(buffer: input, offset: 256, size: 4)],
                      meshBuffers: [.init(group: 1, binding: 0): .init(buffer: output, offset: 256, size: 4)]),
            .copy(source: output, sourceOffset: 256, destination: readback, byteCount: 4),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[42]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif


#if canImport(Metal)
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true), arguments: [false, true])
func metalMeshPrimitiveVaryingReachesFragmentReadback(explicit: Bool) async throws {
    let primitiveAttribute = explicit ? "@per_primitive" : ""
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, \(primitiveAttribute) @location(0) value: f32, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
          if lane == 0u { output.vertex_count = 3u; output.primitive_count = 1u;
            output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
            output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
            output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
            output.primitives[0].value = 0.6;
            output.primitives[0].indices = vec3u(0u, 1u, 2u); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragmentSource = explicit
        ? "enable wgpu_mesh_shader; @fragment fn fs(@location(0) @per_primitive value: f32) -> @location(0) vec4f { return vec4f(value, 0.0, 0.0, 1.0); }"
        : "@fragment fn fs(@location(0) @interpolate(flat) value: f32) -> @location(0) vec4f { return vec4f(value, 0.0, 0.0, 1.0); }"
    let fragment = try compiler.compileRenderStage(.wgsl(fragmentSource), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1)),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        let red = try #require(result.first?.first).littleEndian & 0xff
        #expect((152...153).contains(red))
        #expect(((result.first?.first ?? 0) & 0xff000000) == 0xff000000)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
