import Testing
import SAGACompiler
import SAGATypes
import SAGACore

@Test func vertexAndBindGroupLimitCountsLastAssignedSlots() async throws {
    let caps = try DeviceCapabilities(maxBufferSize: 4096, maxTextureDimension2D: 32,
                                      maxBindGroupsPlusVertexBuffers: 6)
    let backend = RenderTestBackend(capabilities: caps)
    let device = try await Device.make(backend: backend)
    let compiler = IndependentShaderCompiler()
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn f() -> @location(0) vec4f { return vec4f(1.0); }"), entryPoint: "f", to: .msl)
    let layouts: [VertexBufferLayout] = [.init(stride: 0, attributes: []),
        .init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let target = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm, usage: .renderAttachment))
    let vertices = try await device.createBuffer(.init(size: 24, usage: .vertex))
    let uniform = try await device.createBuffer(.init(size: 16, usage: .uniform))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    await backend.release.release()
    for group: UInt32 in [0, 3] {
        let source = """
        @group(\(group)) @binding(0) var<uniform> value: vec4f;
        @vertex fn v(@location(0) xy: vec2f) -> @builtin(position) vec4f { return vec4f(xy + value.xy, 0.0, 1.0); }
        """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "v", to: .msl)
        if group == 3 {
            await #expect(throws: GPUError.self) {
                try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                    vertexBuffers: layouts + [.init(stride: 0, attributes: [])])
            }
        }
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
        let bindings = [ComputeBindingSlot(group: group, binding: 0): ComputeBufferBinding(buffer: uniform)]
        let valid = try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
            vertexBuffers: [.init(buffer: vertices, slot: 1)], uniforms: bindings)])
        _ = try await valid.wait()
        if group == 3 {
            await #expect(throws: GPUError.self) {
                try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices, slot: 1), .init(buffer: vertices, slot: 2)], uniforms: bindings)])
            }
        } else {
            let extra = try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices, slot: 1), .init(buffer: vertices, slot: 4)], uniforms: bindings)])
            _ = try await extra.wait()
            await #expect(throws: GPUError.self) {
                try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices, slot: 1), .init(buffer: vertices, slot: 5)], uniforms: bindings)])
            }
        }
    }
    await device.close()
}
