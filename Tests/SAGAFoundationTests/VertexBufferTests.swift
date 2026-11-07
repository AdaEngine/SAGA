import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

let vertexBufferShader = ShaderInput.wgsl("""
    struct Input {
        @location(4) position: vec2f,
        @builtin(vertex_index) id: u32,
        @location(1) color: vec3f,
    }
    struct Output {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
    }
    @vertex fn v(input: Input) -> Output {
        if input.id >= 3u {
            return Output(vec4f(0.0, 0.0, 0.0, 1.0), input.color);
        }
        return Output(vec4f(input.position, 0.25, 1.0), input.color);
    }
    @fragment fn f(@location(0) color: vec3f) -> @location(0) vec4f {
        return vec4f(color, 1.0);
    }
    """)

let vertexBufferLayouts: [VertexBufferLayout] = [
    .init(stride: 12, attributes: [.init(location: 4, offset: 0, format: .float32x2)]),
    .init(stride: 16, attributes: [.init(location: 1, offset: 4, format: .float32x3)]),
]

private let integerVertexShader = ShaderInput.wgsl("""
    @vertex fn v(@location(4) position: vec2f, @location(1) color: vec3u, @location(7) sign: i32)
        -> @builtin(position) vec4f {
        if sign < 0i && (color.y == 1u && color.z == 0u) {
            return vec4f(position, 0.25, 1.0);
        }
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    @fragment fn f() -> @location(0) vec4f { return vec4f(1.0, 0.0, 1.0, 1.0); }
    """)

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsSignedAndUnsignedVertexInputTypes(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(integerVertexShader, entryPoint: "v", to: format)
    #expect(vertex.vertexInputs.map(\.scalar) == [.u32, .f32, .i32])
    guard case .text(_, let code) = vertex.artifact else { Issue.record("Expected text shader"); return }
    if format == .glsl {
        #expect(code.contains("in uvec3 saga_attribute_1"))
        #expect(code.contains("in int saga_attribute_7"))
        _ = try compiler.parseAndValidate(.glsl(code, stage: .vertex, defines: [:]))
    } else if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(code))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsVertexLocationsAlongsideVertexIndex(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let stage = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "v", to: format)
    #expect(stage.vertexInputs.map(\.location) == [1, 4])
    #expect(stage.vertexInputs.map(\.componentCount) == [3, 2])
    guard case .text(_, let code) = stage.artifact else { Issue.record("Expected text shader"); return }
    switch format {
    case .msl:
        #expect(code.contains("[[attribute(4)]]"))
        #expect(code.contains("[[vertex_id]]"))
    case .hlsl: #expect(code.contains("TEXCOORD4") && code.contains("SV_VertexID"))
    case .glsl:
        #expect(code.contains("layout(location = 4) in vec2"))
        _ = try compiler.parseAndValidate(.glsl(code, stage: .vertex, defines: [:]))
    case .wgsl:
        #expect(code.contains("@location(4)") && code.contains("@builtin(vertex_index)"))
        _ = try compiler.parseAndValidate(.wgsl(code))
    default: Issue.record("Unexpected target")
    }
}

@Test func vertexLayoutAndFetchRangeAreCheckedBeforeQueueSubmission() async throws {
    let backend = RenderTestBackend()
    let device = try await Device.make(backend: backend)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "v", to: .msl)
    let fragment = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "f", to: .msl)
    for layouts in [
        [],
        [.init(stride: 8, attributes: [.init(location: 4, offset: 0, format: .float32x2)])],
        [.init(stride: 8, attributes: [.init(location: 4, offset: 4, format: .float32x2)]), vertexBufferLayouts[1]],
        [vertexBufferLayouts[0], .init(stride: 14, attributes: [.init(location: 1, offset: 4, format: .float32x3)])],
        [vertexBufferLayouts[0], .init(stride: 16, attributes: [.init(location: 1, offset: 4, format: .uint32x3)])],
        vertexBufferLayouts + [vertexBufferLayouts[0]],
    ] {
        await #expect(throws: GPUError.self) {
            try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                  vertexBuffers: layouts)
        }
    }
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                         vertexBuffers: vertexBufferLayouts)
    let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .renderAttachment))
    let positions = try await device.createBuffer(.init(size: 36, usage: .vertex))
    let colors = try await device.createBuffer(.init(size: 48, usage: .vertex))
    let wrongUsage = try await device.createBuffer(.init(size: 48, usage: .copyDestination))
    for bindings: [VertexBufferBinding] in [
        [],
        [.init(buffer: positions), .init(buffer: wrongUsage)],
        [.init(buffer: positions, offset: 2), .init(buffer: colors)],
        [.init(buffer: positions, offset: 8), .init(buffer: colors)],
        [.init(buffer: positions, offset: UInt64.max - 3), .init(buffer: colors)],
    ] {
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([
                .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1),
                        vertexCount: 3, vertexBuffers: bindings),
            ])
        }
    }
    #expect(await backend.submitted == 0)
    await device.close()
}

#if canImport(Metal)
@Test func metalVertexFetchPreservesIntegerTypesAndDefaultComponents() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(integerVertexShader, entryPoint: "v", to: .msl)
        let fragment = try compiler.compileRenderStage(integerVertexShader, entryPoint: "f", to: .msl)
        let layouts: [VertexBufferLayout] = [
            .init(stride: 8, attributes: [.init(location: 4, offset: 0, format: .float32x2)]),
            .init(stride: 0, attributes: [.init(location: 1, offset: 0, format: .uint32x2),
                                          .init(location: 7, offset: 8, format: .sint32)]),
        ]
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                             vertexBuffers: layouts)
        let positions = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let integers = try await device.createBuffer(.init(size: 12, usage: [.vertex, .copyDestination]))
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let xy: [Float] = [-1, 1, 1, 1, -1, -1]
        let ticket = try await device.submitCommands([
            .write(buffer: positions, offset: 0, words: xy.map(\.bitPattern)),
            .write(buffer: integers, offset: 0, words: [0, 1, .max]),
            .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 1),
                    vertexCount: 3, vertexBuffers: [.init(buffer: positions), .init(buffer: integers)]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let pixels = try #require(try await ticket.wait().first)
        #expect(pixels.first == 0xffff00ff)
        #expect(pixels.last == 0xff000000)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func vertexBuffersDriveMetalGeometryAndColorThroughSAGAQueue() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "v", to: .msl)
        let fragment = try compiler.compileRenderStage(vertexBufferShader, entryPoint: "f", to: .msl)
        let layouts: [VertexBufferLayout] = [
            .init(stride: 16, attributes: [.init(location: 4, offset: 0, format: .float32x3)]),
            .init(stride: 0, attributes: [.init(location: 1, offset: 4, format: .float32x2)]),
        ]
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                             vertexBuffers: layouts)
        let positions = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
        let colors = try await device.createBuffer(.init(size: 12, usage: [.vertex, .copyDestination]))
        let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                           usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 1824, usage: [.mapRead, .copyDestination]))
        let positionWords: [Float] = [99, -1, 1, 42, 99, 1, 1, 42, 99, -1, -1, 42]
        let colorWords: [Float] = [99, 0, 1]
        let ticket = try await device.submitCommands([
            .write(buffer: positions, offset: 0, words: positionWords.map(\.bitPattern)),
            .write(buffer: colors, offset: 0, words: colorWords.map(\.bitPattern)),
            .render(pipeline: pipeline, target: texture, clear: .init(red: 0, green: 0, blue: 1, alpha: 1),
                    vertexCount: 3, vertexBuffers: [.init(buffer: positions, offset: 4), .init(buffer: colors)]),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 1824),
        ])
        try await device.destroyBuffer(positions)
        let pixels = try #require(try await ticket.wait().first)
        #expect(pixels.first == 0xff00ff00)
        #expect(pixels.last == 0xffff0000)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
