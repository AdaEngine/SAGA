import SAGABackend
import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshMultisampleResolves() async throws {
    let taskSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        var<workgroup> task_shared: array<u32, 4>;
        @task @payload(payload) @workgroup_size(4)
        fn task(@builtin(local_invocation_index) lane: u32) -> @builtin(mesh_task_size) vec3u {
            if lane == 0u { payload.value = 7u; }
            workgroupBarrier();
            if lane == 1u { task_shared[0] = payload.value + task_shared[1]; }
            workgroupBarrier();
            if lane == 0u { payload.value = task_shared[0]; }
            return vec3u(1u);
        }
        """)
    let meshSource = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        var<workgroup> mesh_shared: array<u32, 4>;
        @mesh(output) @payload(payload) @workgroup_size(4) fn mesh(@builtin(local_invocation_index) lane: u32) {
            if lane == 1u { mesh_shared[0] = payload.value + mesh_shared[1]; }
            workgroupBarrier();
            if lane == 0u && mesh_shared[0] == 7u { output.vertex_count = 3u; output.primitive_count = 1u;
                output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
                output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
                output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
                output.primitives[0].indices = vec3u(0u, 1u, 2u); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(taskSource, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(meshSource, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        await #expect(throws: GPUError.self) {
            try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm, sampleCount: 2)
        }
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm, sampleCount: 4)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment], sampleCount: 4))
        let resolved = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), resolve: .init(texture: texture))])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), resolve: .init(texture: .init(device: texture.device, serial: UInt64.max)))])
        }
        let wrongSize = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
        let wrongFormat = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: .renderAttachment))
        let wrongSamples = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment, sampleCount: 4))
        for invalid in [wrongSize, wrongFormat, wrongSamples] {
            await #expect(throws: GPUError.self) {
                try await device.submitCommands([.drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), resolve: .init(texture: invalid))])
            }
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.drawMesh(pipeline: pipeline, target: resolved, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1))])
        }
        let integerFragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(1u); }"), entryPoint: "fs", to: .msl)
        let integerPipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: integerFragment, colorFormat: .rgba8Uint, sampleCount: 4)
        let integerMSAA = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: .renderAttachment, sampleCount: 4))
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([.drawMesh(pipeline: integerPipeline, target: integerMSAA, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), resolve: .init(texture: wrongFormat))])
        }
        let maskedFragment = try compiler.compileRenderStage(.wgsl("""
            struct Out { @location(0) color: vec4f, @builtin(sample_mask) mask: u32, }
            @fragment fn fs() -> Out { return Out(vec4f(1.0, 0.0, 0.0, 1.0), 0u); }
            """), entryPoint: "fs", to: .msl)
        await #expect(throws: GPUError.self) {
            try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm, alphaToCoverageEnabled: true)
        }
        await #expect(throws: GPUError.self) {
            try await device.createMeshPipeline(task: task, mesh: mesh, fragment: maskedFragment, colorFormat: .rgba8Unorm, sampleCount: 4, alphaToCoverageEnabled: true)
        }
        await #expect(throws: GPUError.self) {
            try await device.createMeshPipeline(task: task, mesh: mesh, fragment: integerFragment, colorFormat: .rgba8Uint, sampleCount: 4, alphaToCoverageEnabled: true)
        }
        let maskedPipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: maskedFragment, colorFormat: .rgba8Unorm, sampleCount: 4)
        let zeroPipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm, sampleCount: 4, sampleMask: 0)
        let alphaPipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm, sampleCount: 4, alphaToCoverageEnabled: true)
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), resolve: .init(texture: resolved)),
            .copyTextureToBuffer(source: resolved, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff0000ff]])
        for (candidate, expected) in [(maskedPipeline, UInt32(0)), (zeroPipeline, UInt32(0)), (alphaPipeline, UInt32(0xff0000ff))] {
            let coverage = try await device.submitCommands([
                .drawMesh(pipeline: candidate, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), resolve: .init(texture: resolved)),
                .copyTextureToBuffer(source: resolved, destination: readback, bytesPerRow: 256),
                .read(buffer: readback, byteCount: 4),
            ]).wait()
            #expect(coverage == [[expected]])
        }
        let depthPipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment,
            colorFormat: .rgba8Unorm, sampleCount: 4, depthState: .init(compare: .less))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
            usage: .renderAttachment, sampleCount: 4))
        let depthResult = try await device.submitCommands([
            .drawMesh(pipeline: depthPipeline, target: texture, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1),
                      resolve: .init(texture: resolved), depth: .init(texture: depth, loadAction: .clear(1))),
            .copyTextureToBuffer(source: resolved, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(depthResult == [[0xff0000ff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
