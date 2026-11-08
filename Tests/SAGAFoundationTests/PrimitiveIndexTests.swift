import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let primitiveIndexShader = """
    enable primitive_index;
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) tint: f32,
    }

    @vertex fn vs(@location(0) point: vec2f, @builtin(instance_index) instance: u32) -> VertexOutput {
        return VertexOutput(vec4f(point.x, point.y - f32(instance), 0.0, 1.0), 1.0);
    }

    struct FragmentInput {
        @location(0) tint: f32,
        @builtin(primitive_index) index: u32,
        @builtin(position) pixel: vec4f,
        @builtin(front_facing) front: bool,
    }

    fn encode(index: u32, tint: f32) -> vec4u {
        return vec4u(index + u32(tint), 0u, 0u, 255u);
    }

    @fragment fn fs(input: FragmentInput) -> @location(0) vec4u {
        return encode(input.index, input.tint);
    }

    @fragment fn fs_direct(@builtin(primitive_index) index: u32) -> @location(0) vec4u {
        return encode(index, 1.0);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func primitiveIndexLowersFragmentBuiltinInDirectAndStructuredInputs(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(primitiveIndexShader)
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
    #expect(!vertex.readsPrimitiveIndex)
    for name in ["fs", "fs_direct"] {
        let fragment = try compiler.compileRenderStage(input, entryPoint: name, to: format)
        #expect(fragment.readsPrimitiveIndex)
        #expect(fragment.fragmentInputBuiltinSlots == (name == "fs" ? 2 : 1))
        #expect(fragment.varyings.map(\.location) == (name == "fs" ? [0] : []))
        guard case .text(_, let output) = fragment.artifact else {
            Issue.record("Expected text artifact")
            continue
        }
        switch format {
        case .wgsl:
            #expect(output.contains("enable primitive_index;"))
            #expect(output.contains("@builtin(primitive_index)"))
            _ = try compiler.parseAndValidate(.wgsl(output), capabilities: [.primitiveIndex])
        case .msl: #expect(output.contains("[[primitive_id]]"))
        case .hlsl: #expect(output.contains("SV_PrimitiveID"))
        case .glsl: #expect(output.contains("uint(gl_PrimitiveID)"))
        default: Issue.record("Unexpected target")
        }
    }
}

@Test func primitiveIndexRequiresTheExtensionAndCapability() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(primitiveIndexShader)) }
    let withoutExtension = String(primitiveIndexShader.dropFirst("enable primitive_index;\n".count))
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl(withoutExtension), capabilities: [.primitiveIndex])
    }
}

@Test(arguments: [
    "@vertex fn vs(@builtin(primitive_index) index: u32) -> @builtin(position) vec4f { return vec4f(f32(index)); }",
    "@fragment fn fs() -> @builtin(primitive_index) u32 { return 0u; }",
    "@fragment fn fs(@builtin(primitive_index) index: i32) -> @location(0) vec4f { return vec4f(f32(index)); }",
    """
    @fragment fn fs(@builtin(primitive_index) a: u32, @builtin(primitive_index) b: u32) -> @location(0) vec4f {
        return vec4f(f32(a + b));
    }
    """,
])
func primitiveIndexRejectsInvalidStageDirectionTypeAndDuplicates(source: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl("enable primitive_index;\n" + source),
                                                         capabilities: [.primitiveIndex])
    }
}

#if canImport(Metal)
@Test func primitiveIndexConsumesAnInputSlotWithoutLoweringTheMaximumLocation() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.primitiveIndex) else {
        await device.close()
        return
    }
    let compiler = IndependentShaderCompiler()
    func stages(_ locations: [Int]) throws -> (RenderStageArtifact, RenderStageArtifact) {
        let varyings = locations.map { "@location(\($0)) value_\($0): f32," }.joined(separator: "\n")
        let source = ShaderInput.wgsl("""
            enable primitive_index;
            struct Varyings { \(varyings) }
            struct VertexOutput { @builtin(position) position: vec4f, \(varyings) }
            @vertex fn vs() -> VertexOutput { return VertexOutput(); }
            @fragment fn fs(values: Varyings, @builtin(primitive_index) index: u32,
                            @builtin(front_facing) front: bool) -> @location(0) vec4f {
                return vec4f(select(values.value_15, f32(index), front));
            }
            """)
        return (try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
                try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl))
    }
    let (validVertex, validFragment) = try stages(Array(0..<13) + [15])
    _ = try await device.createRenderPipeline(vertex: validVertex, fragment: validFragment, colorFormat: .rgba8Unorm)
    let (invalidVertex, invalidFragment) = try stages(Array(0..<14) + [15])
    await #expect(throws: GPUError.validation("Fragment inputs exceed the inter-stage limit")) {
        try await device.createRenderPipeline(vertex: invalidVertex, fragment: invalidFragment, colorFormat: .rgba8Unorm)
    }
    await device.close()
}

@Test func primitiveIndicesResetPerInstanceAndIgnoreStripRestartOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(primitiveIndexShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    guard device.capabilities.features.contains(.primitiveIndex) else {
        await #expect(throws: GPUError.validation("Feature not advertised")) {
            try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                                                   vertexBuffers: [layout])
        }
        await device.close()
        return
    }
    let list = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                                                     vertexBuffers: [layout])
    let directFragment = try compiler.compileRenderStage(source, entryPoint: "fs_direct", to: .msl)
    let strip = try await device.createRenderPipeline(vertex: vertex, fragment: directFragment, colorFormat: .rgba8Uint,
        vertexBuffers: [layout], primitive: .init(topology: .triangleStrip, stripIndexFormat: .uint16))
    let positions = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 16, usage: [.index, .copyDestination]))
    let target = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Uint,
                                                       usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 1024, usage: [.copyDestination, .mapRead]))
    let streams = [VertexBufferBinding(buffer: positions)]
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 1024),
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: positions, offset: 0,
               words: [Float(-1), 0, 0, 0, -1, 1, 0, 0, 1, 0, 1, 1].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 0xffff0002, 0x00040003, 0x00000005]),
        .render(pipeline: list, target: target, clear: .transparentBlack, vertexCount: 6,
                vertexBuffers: streams, instances: .init(count: 2)),
    ] + capture + [
        .renderIndexed(pipeline: strip, target: target, clear: .transparentBlack,
                       indexBuffer: .init(buffer: indices, format: .uint16),
                       draw: .init(indexCount: 7, instances: .init(count: 2)), vertexBuffers: streams),
    ] + capture)
    let words = try await ticket.wait()
    #expect(words.count == 2)
    for pixels in words {
        #expect([pixels[64], pixels[67], pixels[192], pixels[195]] == [0xff000001, 0xff000002, 0xff000001, 0xff000002])
    }
    await device.close()
}
#endif
