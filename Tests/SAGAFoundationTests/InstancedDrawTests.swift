import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let instanceShader = ShaderInput.wgsl("""
    struct Output {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
    }
    @vertex fn v(@location(0) position: vec2f, @location(1) offset: f32, @location(2) red: f32,
                 @builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Output {
        if vertex == 0u { return Output(vec4f(0.0, 0.0, 0.0, 1.0), vec3f(0.0)); }
        var green = 0.0;
        if instance == 2u { green = 1.0; }
        return Output(vec4f(position.x + offset, position.y, 0.25, 1.0), vec3f(red, green, 0.0));
    }
    @fragment fn f(@location(0) color: vec3f) -> @location(0) vec4f { return vec4f(color, 1.0); }
    """)

private let instanceLayouts: [VertexBufferLayout] = [
    .init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)]),
    .init(stride: 8, stepMode: .instance, attributes: [
        .init(location: 1, offset: 0, format: .float32),
        .init(location: 2, offset: 4, format: .float32),
    ]),
]

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func vertexShaderUsesTargetBaseInstanceSemantics(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let stage = try compiler.compileRenderStage(instanceShader, entryPoint: "v", to: format)
    guard case .text(_, let code) = stage.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .msl:
        #expect(code.contains("[[instance_id]]"))
    case .hlsl:
        #expect(stage.hlslProfile == "vs_6_8")
        #expect(code.contains("SV_StartInstanceLocation") && code.contains("SV_StartVertexLocation"))
        #expect(code.contains("(saga_instance + saga_first_instance)"))
    case .glsl:
        #expect(code.hasPrefix("#version 460 core\n"))
        #expect(code.contains("uint(gl_InstanceID) + uint(gl_BaseInstance)"))
        _ = try compiler.parseAndValidate(.glsl(code, stage: .vertex, defines: [:]))
    case .wgsl:
        #expect(code.contains("@builtin(instance_index)"))
        _ = try compiler.parseAndValidate(.wgsl(code))
    default: Issue.record("Unexpected format")
    }
}

@Test func glslBaseInstanceRequiresDesktop460() {
    let source = """
        #version 450 core
        void main() { gl_Position = vec4(float(gl_BaseInstance), 0.0, 0.0, 1.0); }
        """
    #expect(throws: ShaderFailure.self) {
        _ = try IndependentShaderCompiler().parseAndValidate(.glsl(source, stage: .vertex, defines: [:]))
    }
}

private func instancePipeline(_ device: Device) async throws -> RenderPipeline {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(instanceShader, entryPoint: "v", to: .msl)
    let fragment = try compiler.compileRenderStage(instanceShader, entryPoint: "f", to: .msl)
    return try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                 vertexBuffers: instanceLayouts)
}

@Test func validatesInstanceAndVertexRangesBeforeQueueSubmission() async throws {
    let backend = RenderTestBackend()
    let device = try await Device.make(backend: backend)
    let pipeline = try await instancePipeline(device)
    let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .renderAttachment))
    let vertices = try await device.createBuffer(.init(size: 32, usage: .vertex))
    let instances = try await device.createBuffer(.init(size: 16, usage: .vertex))
    let indices = try await device.createBuffer(.init(size: 12, usage: .index))
    let bindings: [VertexBufferBinding] = [.init(buffer: vertices), .init(buffer: instances)]
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    for command in [
        BufferCommand.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: bindings,
                             firstVertex: 1, instances: .init(count: 2, first: 1)),
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: bindings,
                firstVertex: 2, instances: .init(count: 1)),
        .renderIndexed(pipeline: pipeline, target: target, clear: clear, indexBuffer: .init(buffer: indices, format: .uint32),
                       draw: .init(indexCount: 3, instances: .init(count: 2, first: 1)), vertexBuffers: bindings),
        .renderIndexed(pipeline: pipeline, target: target, clear: clear, indexBuffer: .init(buffer: indices, format: .uint32),
                       draw: .init(indexCount: 3, instances: .init(count: .max, first: .max)), vertexBuffers: bindings),
    ] {
        await #expect(throws: GPUError.self) { try await device.submitCommands([command]) }
    }
    #expect(await backend.submitted == 0)
    await device.close()
}

#if canImport(Metal)
@Test func indexedAndNonindexedInstancesUseTheSameOffsetsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await instancePipeline(device)
        let vertices = try await device.createBuffer(.init(size: 32, usage: [.vertex, .copyDestination]))
        let instanceData = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.mapRead, .copyDestination]))
        let bindings: [VertexBufferBinding] = [.init(buffer: vertices), .init(buffer: instanceData)]
        let instances = DrawInstances(count: 2, first: 1)
        let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let positions: [Float] = [99, 99, -0.45, 0.8, 0.45, 0.8, -0.45, -0.8]
        let colors: [Float] = [99, 1, -0.5, 1, 0.5, 0]
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: positions.map(\.bitPattern)),
            .write(buffer: instanceData, offset: 0, words: colors.map(\.bitPattern)),
            .write(buffer: indices, offset: 0, words: [2, 3, 4]),
            .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: bindings,
                    firstVertex: 1, instances: instances),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                           indexBuffer: .init(buffer: indices, format: .uint32),
                           draw: .init(indexCount: 3, baseVertex: -1, instances: instances), vertexBuffers: bindings),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let pixels = try await ticket.wait()
        try #require(pixels.count == 2)
        #expect(pixels[0] == pixels[1])
        #expect(pixels[0][64] == 0xff0000ff)
        #expect(pixels[0][68] == 0xff00ff00)
        #expect(pixels[0].last == 0xffff0000)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
