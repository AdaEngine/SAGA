import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let halfPrecisionRenderSource = """
    enable f16;
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) tint: vec2h,
    }
    struct Colors {
        @location(0) red: f16,
        @location(1) rgba: vec4h,
    }
    struct VertexInput {
        @location(0) data: vec4h,
        @builtin(vertex_index) index: u32,
    }
    struct FragmentInput {
        @builtin(front_facing) front: bool,
        @location(0) tint: vec2h,
    }
    struct ColorValue { @location(0) rgba: vec4h, }

    @vertex fn vs(input: VertexInput) -> VertexOutput {
        let snapshot = input;
        let vertex = snapshot.data;
        return VertexOutput(vec4f(vec2f(vertex.xy), 0.0, 1.0), vertex.zw);
    }
    fn forward(input: FragmentInput) -> FragmentInput { return input; }
    @fragment fn fs(input: FragmentInput) -> Colors {
        var snapshot = forward(input);
        snapshot.tint.x += 0.5h;
        let alpha = select(0.0h, 1.0h, snapshot.front == input.front);
        return Colors(input.tint.x, vec4h(input.tint, snapshot.tint.x, alpha));
    }
    @fragment fn fsSingle(input: FragmentInput) -> @location(0) vec2h {
        var snapshot = input;
        snapshot.tint.y = 0.125h;
        return snapshot.tint;
    }
    @fragment fn fsSameType(input: ColorValue) -> ColorValue { return input; }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func halfPrecisionRenderIOCompiles(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(halfPrecisionRenderSource)
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: format)
    let direct = try compiler.compileRenderStage(input, entryPoint: "fsSingle", to: format)
    let sameType = try compiler.compileRenderStage(input, entryPoint: "fsSameType", to: format)
    #expect(vertex.vertexInputs.first?.scalar == .f16)
    #expect(vertex.varyings == fragment.varyings)
    #expect(fragment.varyings.first?.scalar == .f16)
    #expect(fragment.colorOutputKinds == [0: .float, 1: .float])
    #expect(fragment.colorOutputComponentCounts == [0: 1, 1: 4])
    #expect(direct.colorOutputComponentCounts == [0: 2])
    #expect(sameType.colorOutputComponentCounts == [0: 4])
    for shader in [vertex, fragment, direct, sameType] {
        #expect(shader.requiresShaderF16)
        if case .text(.wgsl, let code) = shader.artifact {
            _ = try compiler.parseAndValidate(.wgsl(code), capabilities: [.shaderFloat16])
        }
    }
    if format == .hlsl { #expect(fragment.hlslProfile == "ps_6_2") }
}

@Test func halfPrecisionRenderIOUsesPackedVertexAttributeSizes() throws {
    #expect(VertexFormat.float16.byteSize == 2)
    #expect(VertexFormat.float16x2.byteSize == 4)
    #expect(VertexFormat.float16x4.byteSize == 8)
    let limits = try DeviceCapabilities(maxBufferSize: 1024)
    _ = try VertexBufferLayout.validate([
        .init(stride: 4, attributes: [.init(location: 0, offset: 2, format: .float16)]),
    ], against: limits)
    #expect(throws: GPUError.self) {
        try VertexBufferLayout.validate([
            .init(stride: 8, attributes: [.init(location: 0, offset: 2, format: .float16x2)]),
        ], against: limits)
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func halfPrecisionRenderIOPreservesNativeHalfValuesOnMetal(_ pulling: Bool) async throws {
    try await checkHalfPrecisionRenderIOOnMetal(.wgsl(halfPrecisionRenderSource), pulling: pulling)
}

@Test func spirvHalfCodeGeneratorRoundTripsRenderLocationsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(halfPrecisionRenderSource)
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    try #require(words.count > 5)
    var capabilities: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        if words[offset] & 0xffff == 17 && count == 2 { capabilities.insert(words[offset + 1]) }
        offset += count
    }
    #expect(capabilities.isSuperset(of: [9, 4436])) // Float16 and StorageInputOutput16
    let input = ShaderInput.spirv(words)
    for format: ShaderOutputFormat in [.msl, .wgsl, .hlsl, .glsl] {
        let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
        let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: format)
        #expect(vertex.vertexInputs.first?.scalar == .f16)
        #expect(vertex.varyings == fragment.varyings && fragment.varyings.first?.scalar == .f16)
        #expect(fragment.colorOutputComponentCounts == [0: 1, 1: 4])
        #expect(vertex.requiresShaderF16 && fragment.requiresShaderF16)
        if format == .wgsl {
            for shader in [vertex, fragment] {
                guard case .text(.wgsl, let code) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
                _ = try compiler.parseAndValidate(.wgsl(code), capabilities: [.shaderFloat16])
            }
        }
    }
    try await checkHalfPrecisionRenderIOOnMetal(input, pulling: false)
}

private func checkHalfPrecisionRenderIOOnMetal(_ input: ShaderInput, pulling: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float16x4)])
    let vertex = pulling
        ? try compiler.compileMetalVertexStage(input, entryPoint: "vs", vertexBuffers: [layout])
        : try compiler.compileRenderStage(input, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .r16Float, vertexBuffers: [layout],
            additionalColorTargets: [1: .init(format: .rgba16Float)])
        let positions = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let red = try await device.createTexture(.init(width: 1, height: 1, format: .r16Float,
                                                       usage: [.renderAttachment, .copySource]))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba16Float,
                                                         usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.mapRead, .copyDestination]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        let snapshots = try await device.submitCommands([
            .write(buffer: positions, offset: 0, words: [
                0xbc00bc00, 0x3a003400, 0xbc004200, 0x3a003400, 0x4200bc00, 0x3a003400,
            ]),
            .render(pipeline: pipeline, target: red, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: positions)],
                    additionalColors: [1: .init(texture: color, loadAction: .clear(clear))]),
            .copyTextureToBuffer(source: red, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[0x3400], [0x3a003400, 0x3c003a00]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
