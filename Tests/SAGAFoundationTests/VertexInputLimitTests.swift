import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func vertexInputLimitShader(location: UInt32) -> ShaderInput {
    .wgsl("""
        struct Raster {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(flat) tag: u32,
        }
        @vertex fn vs(@location(0) point: vec2f, @location(\(location)) tag: u32) -> Raster {
            return Raster(vec4f(point, 0, 1), tag);
        }
        @fragment fn fs(in: Raster) -> @location(0) vec4u {
            return vec4u(in.tag, 0, 0, 255);
        }
        """)
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func vertexInputLimitsTranslateHighLocations(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(vertexInputLimitShader(location: 30), entryPoint: "vs", to: format)
    #expect(vertex.vertexInputs.map(\.location) == [0, 30])
    guard case .text(let actual, let text) = vertex.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    switch format {
    case .wgsl:
        #expect(text.contains("@location(30)"))
        _ = try compiler.parseAndValidate(.wgsl(text))
    case .msl: #expect(text.contains("[[attribute(30)]]"))
    case .hlsl: #expect(text.contains("TEXCOORD30"))
    case .glsl: #expect(text.contains("layout(location = 30)"))
    default: Issue.record("Expected a text target")
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func vertexInputLimitsKeepTheNativeMSLAttributeBound(_ format: ShaderOutputFormat) throws {
    let input = vertexInputLimitShader(location: 31)
    let compiler = IndependentShaderCompiler()
    if format == .msl {
        #expect(throws: ShaderFailure.self) { try compiler.compileRenderStage(input, entryPoint: "vs", to: format) }
    } else {
        let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
        #expect(vertex.vertexInputs.map(\.location) == [0, 31])
    }
}

@Test func vertexInputLimitsRejectLayoutsBeforePipelineAllocation() async throws {
    let limits = try DeviceCapabilities(maxBufferSize: 4096, maxVertexAttributes: 30,
                                        maxVertexBuffers: 1, maxVertexBufferArrayStride: 4096)
    let host = RenderTestBackend(capabilities: limits)
    let device = try await Device.make(backend: host)
    let compiler = IndependentShaderCompiler()
    let source = vertexInputLimitShader(location: 29)
    let vertex = try compiler.compileWebGPURenderStage(source, entryPoint: "vs")
    let fragment = try compiler.compileWebGPURenderStage(source, entryPoint: "fs")
    let point = VertexAttribute(location: 0, offset: 0, format: .float32x2)
    let tag = VertexAttribute(location: 29, offset: 8, format: .uint32)
    do {
        let invalid: [[VertexBufferLayout]] = [
            [.init(stride: 12, attributes: [point, .init(location: 30, offset: 8, format: .uint32)])],
            [.init(stride: 12, attributes: [point]), .init(stride: 12, attributes: [tag])],
            [.init(stride: 4100, attributes: [point, tag])],
            [.init(stride: 0, attributes: [point, .init(location: 29, offset: 4096, format: .uint32)])],
        ]
        for layout in invalid {
            await #expect(throws: GPUError.self) {
                try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                                                      vertexBuffers: layout)
            }
        }
        #expect(await host.allocatedRenderPipelines == 0)
        for stride: UInt32 in [4096, 0] {
            let lastTag = VertexAttribute(location: 29, offset: 4092, format: .uint32)
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                vertexBuffers: [.init(stride: stride, attributes: [point, lastTag])])
        }
        #expect(await host.allocatedRenderPipelines == 2)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

#if canImport(Metal)
@Test func vertexInputLimitsFetchLocation30OnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = vertexInputLimitShader(location: 30)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let layout = VertexBufferLayout(stride: 12, attributes: [
            .init(location: 0, offset: 0, format: .float32x2),
            .init(location: 30, offset: 8, format: .uint32),
        ])
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                                                             vertexBuffers: [layout])
        let positions = try await device.createBuffer(.init(size: 36, usage: [.vertex, .copyDestination]))
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                            usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let lo = Float(-1).bitPattern
        let hi = Float(3).bitPattern
        let result = try await device.submitCommands([
            .write(buffer: positions, offset: 0, words: [lo, lo, 77, hi, lo, 77, lo, hi, 77]),
            .render(pipeline: pipeline, target: texture, clear: .transparentBlack, vertexCount: 3,
                    vertexBuffers: [.init(buffer: positions)]),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff00004d]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
