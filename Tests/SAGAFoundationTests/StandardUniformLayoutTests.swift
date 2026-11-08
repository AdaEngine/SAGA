import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let standardUniformDeclarations = """
    requires uniform_buffer_standard_layout;
    requires unrestricted_pointer_parameters;
    struct Payload { bias: u32, }
    struct Settings { compact: array<vec2u, 3>, scale: Payload, tail: u32, }
    @group(0) @binding(0) var<uniform> settings: Settings;
    @group(0) @binding(2) var<uniform> dense: array<u32, 4>;
    override increment = 1u;

    """

private let standardUniformCompute = """
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;
    fn readDense(values: ptr<uniform, array<u32, 4>>, lane: u32) -> u32 { return (*values)[lane]; }
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let config = settings;
        let item = config.compact[i32(lane) - 1i];
        output[lane] = vec4u(item, config.scale.bias + readDense(&dense, lane), config.tail + increment);
    }
    """

private let standardUniformRender = """
    @vertex fn vertexMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
        let reach = f32(settings.compact[1].x);
        if id == 0u { return vec4f(-1.0f, -1.0f, 0.0f, 1.0f); }
        if id == 1u { return vec4f(reach, -1.0f, 0.0f, 1.0f); }
        return vec4f(-1.0f, reach, 0.0f, 1.0f);
    }
    @fragment fn fragmentMain() -> @location(0) vec4f {
        let valid = settings.scale.bias == 7u && settings.tail + increment == 12u;
        return select(vec4f(1.0f, 0.0f, 0.0f, 1.0f), vec4f(0.0f, 1.0f, 0.0f, 1.0f), valid);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesStandardUniformLayout(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let compute = try compiler.compileCompute(.wgsl(standardUniformDeclarations + standardUniformCompute),
                                               to: format, constants: ["increment": 3])
    #expect(compute.uniformBindings.map(\.minimumByteSize) == [32, 16])
    #expect(compute.requiresUniformBufferStandardLayout == (format == .wgsl))
    let render = ShaderInput.wgsl(standardUniformDeclarations + standardUniformRender)
    let vertex = try compiler.compileRenderStage(render, entryPoint: "vertexMain", to: format, constants: ["increment": 3])
    let fragment = try compiler.compileRenderStage(render, entryPoint: "fragmentMain", to: format, constants: ["increment": 3])
    #expect(vertex.uniformBindings.map(\.minimumByteSize) == [32])
    #expect(fragment.uniformBindings.map(\.minimumByteSize) == [32])
    #expect(vertex.requiresUniformBufferStandardLayout == (format == .wgsl))
    #expect(fragment.requiresUniformBufferStandardLayout == (format == .wgsl))
    for artifact in [compute.artifact, vertex.artifact, fragment.artifact] {
        guard case .text(_, let generated) = artifact else { Issue.record("Expected source"); return }
        if format == .wgsl {
            #expect(generated.contains("requires uniform_buffer_standard_layout;"))
            _ = try compiler.parseAndValidate(.wgsl(generated))
        } else if format == .glsl {
            #expect(generated.contains("uvec4 words["))
        }
    }
}

@Test func rejectsStandardUniformLayoutWithoutDirective() {
    let directive = "requires uniform_buffer_standard_layout;"
    let declarations = String(standardUniformDeclarations.dropFirst(directive.count))
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(declarations + standardUniformCompute))
    }
}

@Test(arguments: ["array<u32>", "atomic<u32>", "bool"])
func standardUniformLayoutStillRequiresFixedHostShareableData(type: String) {
    let source = ShaderInput.wgsl("""
        requires uniform_buffer_standard_layout;
        @group(0) @binding(0) var<uniform> value: \(type);
        @compute @workgroup_size(1) fn run() {}
        """)
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(source) }
}

#if canImport(Metal)
@Test func readsCompactStandardUniformLayoutsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let computeShader = try compiler.compileCompute(.wgsl(standardUniformDeclarations + standardUniformCompute),
                                                     to: .msl, constants: ["increment": 3])
    let renderShader = ShaderInput.wgsl(standardUniformDeclarations + standardUniformRender)
    let vertex = try compiler.compileRenderStage(renderShader, entryPoint: "vertexMain", to: .msl, constants: ["increment": 3])
    let fragment = try compiler.compileRenderStage(renderShader, entryPoint: "fragmentMain", to: .msl, constants: ["increment": 3])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let webShader = try compiler.compileCompute(.wgsl(standardUniformDeclarations + standardUniformCompute), to: .wgsl)
        await #expect(throws: GPUError.validation("Feature not advertised")) {
            try await device.createComputePipeline(webShader)
        }
        let webVertex = try compiler.compileRenderStage(renderShader, entryPoint: "vertexMain", to: .wgsl)
        let webFragment = try compiler.compileRenderStage(renderShader, entryPoint: "fragmentMain", to: .wgsl)
        await #expect(throws: GPUError.validation("Feature not advertised")) {
            try await device.createRenderPipeline(vertex: webVertex, fragment: webFragment, colorFormat: .rgba8Unorm)
        }
        let compute = try await device.createComputePipeline(computeShader)
        let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let config = try await device.createBuffer(.init(size: 32, usage: [.uniform, .copyDestination]))
        let dense = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                           usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: config, offset: 0, words: [10, 20, 30, 40, 50, 60, 7, 9]),
            .write(buffer: dense, offset: 0, words: [100, 200, 300, 400]),
            .dispatchCompute(pipeline: compute,
                             bindings: [.init(binding: 0): config, .init(binding: 1): output, .init(binding: 2): dense],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
            .render(pipeline: render, target: target, clear: .init(red: 0, green: 0, blue: 1, alpha: 1),
                    vertexCount: 3, uniforms: [.init(binding: 0): .init(buffer: config)]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 512),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.map(\.count) == [16, 128])
        #expect(snapshots[0] == [0, 0, 107, 12, 10, 20, 207, 12, 30, 40, 307, 12, 50, 60, 407, 12])
        let pixels = snapshots[1]
        #expect([pixels[0], pixels[1], pixels[64], pixels[65]] == [0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
