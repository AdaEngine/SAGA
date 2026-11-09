import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

@Test func metalVertexPullingCompilesLocationsBeyondTheNativeDescriptor() throws {
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@location(44) value: vec4f) -> @builtin(position) vec4f { return value; }
        """)
    let compiler = IndependentShaderCompiler()
    let layout = VertexBufferLayout(stride: 4, attributes: [.init(location: 44, offset: 0, format: .unorm8x4BGRA)])
    let shader = try compiler.compileMetalVertexStage(source, entryPoint: "vs", vertexBuffers: [layout])
    #expect(shader.usesMetalVertexPulling)
    #expect(shader.vertexInputs.map(\.location) == [44])
    guard case .text(.msl, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(code.contains("[[buffer(0)]]") && code.contains("[[buffer(8)]]"))
    #expect(!code.contains("[[attribute(") && !code.contains("[[stage_in]]"))
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalVertexStage(source, entryPoint: "vs", vertexBuffers: [])
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileMetalVertexStage(source, entryPoint: "vs",
            vertexBuffers: [.init(stride: 4, attributes: [.init(location: 44, offset: 0, format: .uint8x4)])])
    }
}

#if canImport(Metal)
@Test func metalVertexPullingUsesBoundOffsetsAndDrawIndices() async throws {
    let source = ShaderInput.wgsl("""
        struct Raster {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(flat) color: vec4u,
        }
        @vertex fn vs(@location(0) point: vec2f, @location(17) color: vec4f, @location(3) tag: vec4u,
                      @builtin(instance_index) i: u32) -> Raster {
            return Raster(vec4f(point, 0, 1),
                          vec4u(u32(round(color.x * 255.0)), tag.x + tag.y, tag.z + tag.w + i, 255u));
        }
        @fragment fn fs(input: Raster) -> @location(0) vec4u { return input.color; }
        """)
    let layout: [VertexBufferLayout] = [
        .init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)]),
        .init(stride: 8, stepMode: .instance, attributes: [
            .init(location: 17, offset: 0, format: .unorm8x4BGRA),
            .init(location: 3, offset: 4, format: .uint8x2),
        ]),
    ]
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileMetalVertexStage(source, entryPoint: "vs", vertexBuffers: layout)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        await #expect(throws: GPUError.validation("Vertex buffers do not match the Metal pulling layout")) {
            try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                vertexBuffers: [layout[0], .init(stride: 12, stepMode: .instance, attributes: layout[1].attributes)])
        }
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                                                             vertexBuffers: layout)
        let positions = try await device.createBuffer(.init(size: 40, usage: [.vertex, .copyDestination]))
        let attributes = try await device.createBuffer(.init(size: 32, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                           usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let streams = [VertexBufferBinding(buffer: positions, offset: 8), .init(buffer: attributes, offset: 8)]
        let instances = DrawInstances(first: 2)
        let lo = Float(-1).bitPattern
        let hi = Float(3).bitPattern
        let result = try await device.submitCommands([
            .write(buffer: positions, offset: 0, words: [0, 0, 0, 0, lo, lo, hi, lo, lo, hi]),
            .write(buffer: attributes, offset: 0, words: [0, 0, 0, 0, 0, 0, 0xff402010, 0x00001711]),
            .write(buffer: indices, offset: 0, words: [0x00030002, 4]),
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                    vertexBuffers: streams, firstVertex: 1, instances: instances),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .renderIndexed(pipeline: pipeline, target: target, clear: .transparentBlack,
                           indexBuffer: .init(buffer: indices, format: .uint16),
                           draw: .init(indexCount: 3, baseVertex: -1, instances: instances), vertexBuffers: streams),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff032840], [0xff032840]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
