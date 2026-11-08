import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let invariantPositionShader = """
    struct Raster {
        @location(0) tint: f32,
        @builtin(position) @invariant point: vec4f,
    }

    fn project(value: vec2f) -> vec4f {
        let xy = value * vec2f(0.79, 0.81) + vec2f(0.12, 0.11);
        let z = value.x * 0.123 + value.y * 0.077 + 0.45;
        return vec4f(xy, z, 1.0);
    }

    @vertex fn direct(@location(0) vertex: vec2f) -> @invariant @builtin(position) vec4f {
        return project(vertex);
    }

    @vertex fn structured(@location(0) vertex: vec2f) -> Raster {
        return Raster(1.0, project(vertex));
    }

    @fragment fn fill(@builtin(position) @invariant point: vec4f) -> @location(0) vec4f {
        return vec4f(0.0, select(0.0, 1.0, point.w > 0.0), 0.0, 1.0);
    }

    @fragment fn shade(input: Raster) -> @location(0) vec4f {
        return vec4f(0.0, input.tint, 0.0, 1.0);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func invariantPositionSurvivesDirectAndStructuredVertexOutputs(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(invariantPositionShader)
    for name in ["direct", "structured"] {
        let vertex = try compiler.compileRenderStage(input, entryPoint: name, to: format)
        #expect(vertex.invariantPosition)
        #expect(vertex.varyings.map(\.location) == (name == "structured" ? [0] : []))
        guard case .text(_, let source) = vertex.artifact else { Issue.record("Expected shader text"); continue }
        switch format {
        case .wgsl:
            #expect(source.contains("@invariant @builtin(position)"))
            let module = try compiler.parseAndValidate(.wgsl(source)).module
            let entry = try #require(module.entryPoints.first)
            let function = try module.functions.value(for: entry.function)
            let result = try #require(function.result)
            if case .structure(let members, _) = try module.types.value(for: result.type).kind {
                #expect(members.contains { $0.binding == .builtIn(.position(invariant: true)) })
            } else {
                #expect(result.binding == .builtIn(.position(invariant: true)))
            }
        case .msl: #expect(source.contains("[[position, invariant]]"))
        case .hlsl: #expect(source.contains("precise float4") && source.contains("SV_Position"))
        case .glsl: #expect(source.contains("invariant gl_Position;"))
        default: Issue.record("Unexpected target")
        }
    }
    for name in ["fill", "shade"] {
        let fragment = try compiler.compileRenderStage(input, entryPoint: name, to: format)
        #expect(!fragment.invariantPosition)
        guard case .text(_, let source) = fragment.artifact else { Issue.record("Expected shader text"); continue }
        if format == .wgsl {
            #expect(source.contains("@invariant @builtin(position)"))
            _ = try compiler.parseAndValidate(.wgsl(source))
        } else {
            #expect(!source.contains("invariant") && !source.contains("precise"))
        }
    }
}

@Test(arguments: [
    "@vertex fn vs() -> @builtin(position) @invariant() vec4f { return vec4f(0.0); }",
    "@vertex fn vs() -> @builtin(position) @invariant(1) vec4f { return vec4f(0.0); }",
    "@vertex fn vs() -> @builtin(position) @invariant @invariant vec4f { return vec4f(0.0); }",
    "@fragment fn fs() -> @location(0) @invariant vec4f { return vec4f(0.0); }",
    "@vertex fn vs(@builtin(vertex_index) @invariant index: u32) -> @builtin(position) vec4f { return vec4f(f32(index)); }",
    "struct Output { @invariant value: vec4f, }",
    "fn helper() -> @builtin(position) @invariant vec4f { return vec4f(0.0); }",
])
func invariantPositionRejectsMalformedAttributesAndOtherBindings(source: String) {
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func invariantPositionMatchesDepthAcrossPipelinesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(invariantPositionShader)
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    let direct = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "direct", to: .msl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fill", to: .msl),
        colorFormat: .rgba8Unorm, vertexBuffers: [layout], depthState: .init(compare: .always))
    let structured = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "structured", to: .msl),
        fragment: compiler.compileRenderStage(source, entryPoint: "shade", to: .msl),
        colorFormat: .rgba8Unorm, vertexBuffers: [layout], depthState: .init(compare: .equal, writeEnabled: false))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let color = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 4, height: 4, format: .depth32Float, usage: .renderAttachment))
    let readback = try await device.createBuffer(.init(size: 1024, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 1024),
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: direct, target: color, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)],
                depth: .init(texture: depth, loadAction: .clear(1))),
    ] + capture + [
        .render(pipeline: structured, target: color, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)],
                depth: .init(texture: depth, loadAction: .load)),
    ] + capture)
    let snapshots = try await ticket.wait()
    #expect(snapshots.count == 2)
    for snapshot in snapshots {
        let pixels = (0..<4).flatMap { row in Array(snapshot[(row * 64)..<(row * 64 + 4)]) }
        #expect(pixels.filter { $0 == 0xff00ff00 }.count == 9)
        #expect(pixels.filter { $0 == 0xffff0000 }.count == 7)
    }
    #expect(snapshots.first == snapshots.last)
    await device.close()
}
#endif
