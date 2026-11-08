import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let subgroupGatherShader = """
    enable subgroups;
    struct Inputs {
        @builtin(subgroup_invocation_id) lane: u32,
        @builtin(subgroup_size) size: u32,
    }
    @group(0) @binding(0) var<storage, read_write> out: array<vec4u, 256>;
    fn paired(lane: u32) -> u32 { return subgroupShuffle(lane, lane ^ 1u); }
    @compute @workgroup_size(128) fn run(input: Inputs, @builtin(local_invocation_index) index: u32) {
        let lane = input.lane;
        let size = input.size;
        let down = select(lane, subgroupShuffleDown(lane, 1u), lane + 1u < size);
        let up = select(lane, subgroupShuffleUp(lane, 1u), lane != 0u);
        out[index * 2u] = vec4u(lane, size, subgroupBroadcast(lane, 0u), subgroupBroadcastFirst(lane));
        out[index * 2u + 1u] = vec4u(paired(lane), subgroupShuffleXor(lane, size - 1u), down, up);
    }
    """

private let fragmentSubgroupShader = """
    enable subgroups;
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f { return vec4f(point, 0.5, 1.0); }
    @fragment fn fs(@builtin(subgroup_size) size: u32, @builtin(subgroup_invocation_id) lane: u32) -> @location(0) vec4u {
        return vec4u(size, lane, subgroupBroadcastFirst(lane), 255u);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func subgroupGatherAndBuiltinsLowerInComputeAndFragment(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let compute = try compiler.compileCompute(.wgsl(subgroupGatherShader), to: format)
    let fragment = try compiler.compileRenderStage(.wgsl(fragmentSubgroupShader), entryPoint: "fs", to: format)
    #expect(compute.requiresSubgroups)
    #expect(fragment.requiresSubgroups)
    let vertex = try compiler.compileRenderStage(.wgsl(fragmentSubgroupShader), entryPoint: "vs", to: format)
    #expect(!vertex.requiresSubgroups)
    guard case .text(_, let computeSource) = compute.artifact,
          case .text(_, let fragmentSource) = fragment.artifact else { Issue.record("Expected shader text"); return }
    switch format {
    case .wgsl:
        for text in [computeSource, fragmentSource] {
            #expect(text.contains("enable subgroups;"))
            _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.subgroup])
        }
    case .msl:
        #expect(computeSource.contains("simd_shuffle("))
        #expect(fragmentSource.contains("simd_broadcast_first("))
    case .hlsl:
        #expect(computeSource.contains("WaveReadLaneAt("))
        #expect(fragmentSource.contains("WaveGetLaneCount()") && fragmentSource.contains("WaveReadLaneFirst("))
    case .glsl:
        #expect(computeSource.contains("GL_KHR_shader_subgroup_shuffle") && computeSource.contains("subgroupShuffleXor("))
        #expect(fragmentSource.contains("gl_SubgroupInvocationID") && fragmentSource.contains("subgroupBroadcastFirst("))
    default: Issue.record("Unexpected format")
    }
}

@Test func subgroupRequiresExplicitWGSLFeatureAndValidGatherOperands() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(subgroupGatherShader)) }
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl(String(subgroupGatherShader.dropFirst("enable subgroups;\n".count))),
                                      capabilities: [.subgroup])
    }
    for expression in ["subgroupBroadcast(lane, lane)", "subgroupBroadcast(lane, 128u)",
                       "subgroupShuffle(lane, -1i)", "subgroupShuffle(true, 0u)", "subgroupShuffleDown(lane, 1i)"] {
        let source = "enable subgroups; @compute @workgroup_size(32) fn run(@builtin(subgroup_invocation_id) lane: u32) { _ = \(expression); }"
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(source), capabilities: [.subgroup]) }
    }
    let divergent = """
        enable subgroups;
        @compute @workgroup_size(32) fn run(@builtin(subgroup_invocation_id) lane: u32) {
            if lane == 0u { _ = subgroupBroadcastFirst(lane); }
        }
        """
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(divergent), capabilities: [.subgroup]) }
}

@Test func subgroupSelectorIsValidatedAfterPipelineSpecialization() throws {
    let source = ShaderInput.wgsl("""
        enable subgroups;
        override base: u32 = 0u;
        @group(0) @binding(0) var<storage, read_write> out: array<u32, 32>;
        @compute @workgroup_size(32) fn run(@builtin(subgroup_invocation_id) lane: u32) {
            out[lane] = subgroupShuffle(lane, base + 1u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileCompute(source, to: .msl, constants: ["base": 1])
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, to: .msl, constants: ["base": 127])
    }
}

@Test func subgroupGatherOfFloat16VectorsRequiresTheGLSLSubgroupTypeExtension() throws {
    let source = ShaderInput.wgsl("""
        enable f16, subgroups;
        @group(0) @binding(0) var<storage, read_write> out: array<vec2h, 32>;
        @compute @workgroup_size(32) fn run(@builtin(subgroup_invocation_id) lane: u32) {
            out[lane] = subgroupShuffle(vec2h(f16(lane), 1.0h), 0u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(source, to: .glsl)
    guard case .text(_, let code) = shader.artifact else { Issue.record("Expected shader text"); return }
    #expect(code.contains("#extension GL_EXT_shader_subgroup_extended_types_float16 : require"))
    #expect(shader.requiresShaderF16 && shader.requiresSubgroups)
}

#if canImport(Metal)
@Test func computeSubgroupGatherReadsNeighborsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.subgroups])
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(subgroupGatherShader), to: .msl))
    let output = try await device.createBuffer(.init(size: 4096, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4096, usage: [.copyDestination, .mapRead]))
    let results = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 4096),
        .read(buffer: readback, byteCount: 4096),
    ]).wait()
    let words = try #require(results.first)
    #expect(words.count == 1024)
    for index in 0..<128 {
        let start = index * 8
        let lane = words[start]
        let size = words[start + 1]
        #expect(size >= 4 && size <= 128 && lane < size)
        #expect(Array(words[(start + 2)..<(start + 8)]) ==
                [0, 0, lane ^ 1, size - 1 - lane, lane + 1 < size ? lane + 1 : lane, lane == 0 ? 0 : lane - 1])
    }
    await device.close()
}

@Test func fragmentSubgroupGatherReportsAnActiveLaneOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.subgroups])
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(fragmentSubgroupShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 512, usage: [.mapRead, .copyDestination]))
    let snapshots = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)]),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
    ]).wait()
    let pixels = try #require(snapshots.first)
    for word in [pixels[0], pixels[1], pixels[64], pixels[65]] {
        let size = word & 255
        let lane = (word >> 8) & 255
        let first = (word >> 16) & 255
        #expect(word >> 24 == 255 && size >= 4 && size <= 128 && lane < size && first <= lane)
    }
    await device.close()
}
#endif
