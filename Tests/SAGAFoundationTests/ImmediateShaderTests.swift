import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let immediateLayoutShader = """
    requires immediate_address_space;
    struct Pair { first: u32, second: u32, }
    struct Settings { axis: vec3u, pair: Pair, last: u32, }
    var<immediate> settings: Settings;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
    fn read_pair() -> u32 { return settings.pair.second; }
    @compute @workgroup_size(1) fn run() {
        if settings.last != 0u { workgroupBarrier(); }
        output[0] = settings.axis.z;
        output[1] = settings.pair.first;
        output[2] = read_pair();
        output[3] = settings.last;
    }
    """

private let renderImmediateShader = ShaderInput.wgsl("""
    requires immediate_address_space;
    struct FragmentData { reserved: vec2f, color: vec4f, }
    var<immediate> vertex_data: vec2f;
    var<immediate> fragment_data: FragmentData;
    @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f {
        return vec4f(position + vertex_data, 0, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return fragment_data.color; }
    """)

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsImmediateDataWithStorageLayout(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(immediateLayoutShader), to: format)
    #expect(shader.uniformBindings.isEmpty)
    #expect(shader.storageBindings.count == 1)
    let layout = try #require(shader.immediateLayout)
    #expect(layout.byteSize == 32)
    guard case .text(_, let text) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(text.contains("requires immediate_address_space;") && text.contains("var<immediate>"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.immediates))
    case .msl:
        #expect(layout.metalBufferIndex == 21)
        #expect(text.contains("constant uint* saga_immediate"))
    case .hlsl:
        #expect(layout.hlslBinding == .init(group: 4, binding: 1))
        #expect(text.contains("uint4 saga_immediate[2]"))
    case .glsl:
        #expect(layout.glslUniformBinding == 33)
        #expect(text.contains("uvec4 words[2]"))
    default: Issue.record("Unexpected format")
    }
}

@Test func immediateDataIsStageLocalInRenderShaders() throws {
    let source = ShaderInput.wgsl("""
        var<immediate> vertex_data: f32;
        var<immediate> fragment_data: vec4f;
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, vertex_data); }
        @fragment fn fs() -> @location(0) vec4f { return fragment_data; }
        """)
    let compiler = IndependentShaderCompiler()
    for (stage, entry) in [(ShaderStage.vertex, "vs"), (.fragment, "fs")] {
        let shader = try compiler.compileRenderStage(source, entryPoint: entry, to: .msl)
        #expect(shader.stage == stage)
        #expect(shader.uniformBindings.isEmpty)
        #expect(shader.immediateLayout?.byteSize == (stage == .vertex ? 4 : 16))
        #expect(shader.immediateLayout?.metalBufferIndex == (stage == .vertex ? 30 : 1))
    }
}

@Test(arguments: [
    "var<immediate> invalid: array<u32, 2>;",
    "struct Inner { values: array<u32, 2>, } struct Outer { inner: Inner, } var<immediate> invalid: Outer;",
    "var<immediate> invalid: atomic<u32>;",
    "var<immediate> invalid: bool;",
    "@group(0) @binding(0) var<immediate> invalid: u32;",
    "var<immediate> invalid: u32 = 0u;",
    "var<immediate> invalid: u32; @compute @workgroup_size(1) fn run() { invalid = 1u; }",
    """
    var<immediate> first: u32;
    var<immediate> second: u32;
    fn read_first() -> u32 { return first; }
    @compute @workgroup_size(1) fn run() { _ = read_first() + second; }
    """,
])
func rejectsInvalidImmediateData(source: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.immediates))
    }
}

@Test func immediateDataRequiresCapability() {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(immediateLayoutShader))
    }
}

#if canImport(Metal)
@Test func renderImmediateDataSharesWordsAcrossStagesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(renderImmediateShader, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(renderImmediateShader, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    #expect(pipeline.immediateByteSize == 32)
    let positions = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let arguments = try await device.createBuffer(.init(size: 36, usage: [.indirect, .copyDestination]))
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 256, usage: [.mapRead, .copyDestination]))
    let streams = [VertexBufferBinding(buffer: positions)]
    let index = IndexBufferBinding(buffer: indices, format: .uint32)
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    for invalid in [[], Array(repeating: UInt32(0), count: 9)] {
        await #expect(throws: GPUError.validation("Render immediate data size does not match the pipeline layout")) {
            try await device.submitCommands([.render(pipeline: pipeline, target: target, clear: clear,
                                                     vertexCount: 3, vertexBuffers: streams, immediates: invalid)])
        }
    }
    func words(_ x: Float, _ r: Float, _ g: Float, _ b: Float) -> [UInt32] {
        [x, 0, -7, -9, r, g, b, 1].map(\.bitPattern)
    }
    let setup: [BufferCommand] = [
        .write(buffer: positions, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .write(buffer: arguments, offset: 0, words: [3, 1, 0, 0, 3, 1, 0, 0, 0]),
    ]
    let draws: [BufferCommand] = [
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                vertexBuffers: streams, immediates: words(0, 1, 0, 0)),
        .renderIndexed(pipeline: pipeline, target: target, clear: clear, indexBuffer: index,
                       draw: .init(indexCount: 3), vertexBuffers: streams, immediates: words(0, 0, 1, 0)),
        .renderIndirect(pipeline: pipeline, target: target, colorLoad: .clear(clear), indirectBuffer: arguments,
                        vertexBuffers: streams, immediates: words(4, 1, 0, 0)),
        .renderIndirect(pipeline: pipeline, target: target, colorLoad: .clear(clear), indirectBuffer: arguments,
                        indirectOffset: 16, indexBuffer: index, vertexBuffers: streams, immediates: words(0, 1, 1, 0)),
    ]
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]
    let snapshots = try await device.submitCommands(setup + draws.flatMap { [$0] + capture }).wait()
    #expect(snapshots == [[0xff00_00ff], [0xff00_ff00], [0xffff_0000], [0xff00_ffff]])
    await device.close()
}

@Test func immediateDataDispatchPreservesPackedOffsetsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(immediateLayoutShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    #expect(pipeline.immediateByteSize == 32)
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let arguments = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .indirect]))
    let bindings: [ComputeBindingSlot: ResourceID] = [.init(binding: 0): output]
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([.dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1))])
    }
    let words = try await device.submitCommands([
        .write(buffer: arguments, offset: 0, words: [1, 1, 1]),
        .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1),
                         immediates: [1, 2, 0x1122_3344, 0xaabb_ccdd, 0x8123_4567, 9, .max, .max]),
        .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
        .dispatchComputeIndirect(pipeline: pipeline, bindings: bindings.mapValues { .init(buffer: $0) },
                                 indirectBuffer: arguments, immediates: [1, 2, 21, 22, 23, 0, .max, .max]),
        .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]).wait()
    #expect(words == [[0x1122_3344, 0xaabb_ccdd, 0x8123_4567, 9], [21, 22, 23, 0]])
    await device.close()
}
#endif
