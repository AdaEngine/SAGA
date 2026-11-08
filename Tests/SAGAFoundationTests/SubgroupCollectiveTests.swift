import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let subgroupCollectiveShader = """
    enable subgroups;
    @group(0) @binding(0) var<storage, read_write> out: array<vec4u, 896>;

    fn sum(lane: u32) -> u32 { return subgroupAdd(lane); }

    @compute @workgroup_size(128)
    fn run(@builtin(subgroup_size) size: u32, @builtin(subgroup_invocation_id) lane: u32,
           @builtin(local_invocation_index) index: u32) {
        let base = index * 7u;
        let factor = select(1u, 2u, lane < 3u);
        let leader = subgroupElect();
        out[base] = vec4u(size, lane, u32(subgroupAll(lane < size)), u32(subgroupAny(lane == size - 1u)));
        out[base + 1u] = subgroupBallot((lane & 1u) == 0u);
        out[base + 2u] = vec4u(sum(lane), subgroupMul(factor), subgroupMin(lane + 7u), subgroupMax(lane));
        out[base + 3u] = vec4u(subgroupAnd(lane | 0x80000000u), subgroupOr(lane),
                              subgroupXor(lane), subgroupInclusiveAdd(lane));
        out[base + 4u] = vec4u(subgroupExclusiveAdd(lane), subgroupExclusiveMul(factor),
                              subgroupInclusiveMul(factor), u32(subgroupAll(lane != 0u)) + u32(leader));
        out[base + 5u] = subgroupBallot();
        let q = quadBroadcast(lane, 0u);
        let x = quadSwapX(lane);
        let y = quadSwapY(lane);
        let d = quadSwapDiagonal(lane);
        out[base + 6u] = vec4u(u32(quadSwapX(x) == lane), u32(quadSwapY(y) == lane),
                              u32(quadSwapY(x) == d), u32(quadSwapDiagonal(q) == q));
    }
    """

private let subgroupCollectiveFragment = """
    enable subgroups;
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f { return vec4f(point, 0.5, 1.0); }
    @fragment fn fs(@builtin(position) point: vec4f) -> @location(0) vec4u {
        let pixel = u32(point.x) + 2u * u32(point.y);
        let visible = u32(subgroupAny(pixel < 4u));
        return vec4u(quadBroadcast(visible, 0u), quadSwapX(pixel), quadSwapY(pixel), quadSwapDiagonal(pixel));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func subgroupCollectiveOperationsLowerToEveryTextTarget(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let compute = try compiler.compileCompute(.wgsl(subgroupCollectiveShader), to: format)
    let fragment = try compiler.compileRenderStage(.wgsl(subgroupCollectiveFragment), entryPoint: "fs", to: format)
    #expect(compute.requiresSubgroups && fragment.requiresSubgroups)
    guard case .text(_, let text) = compute.artifact, case .text(_, let fragmentText) = fragment.artifact else {
        Issue.record("Expected shader text")
        return
    }
    switch format {
    case .wgsl:
        #expect(text.contains("enable subgroups;") && text.contains("subgroupBallot(true)"))
        for shader in [text, fragmentText] {
            _ = try compiler.parseAndValidate(.wgsl(shader), capabilities: [.subgroup])
        }
    case .msl:
        #expect(text.contains("simd_ballot(") && text.contains(">> 32"))
        #expect(text.contains("simd_prefix_exclusive_product(") && fragmentText.contains("simd_any("))
        #expect(fragmentText.contains("quad_shuffle_xor("))
    case .hlsl:
        #expect(text.contains("WaveActiveBallot(") && text.contains("WavePrefixProduct("))
        #expect(fragmentText.contains("WaveActiveAnyTrue(") && fragmentText.contains("QuadReadAcrossDiagonal("))
    case .glsl:
        #expect(text.contains("GL_KHR_shader_subgroup_arithmetic") && text.contains("GL_KHR_shader_subgroup_ballot"))
        #expect(fragmentText.contains("GL_KHR_shader_subgroup_vote") && fragmentText.contains("GL_KHR_shader_subgroup_quad"))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: ["subgroupAll(1u)", "subgroupBallot(1u)", "subgroupAnd(1.0f)", "subgroupAdd(true)",
                  "subgroupExclusiveAdd(1u, 2u)", "quadBroadcast(1u, 4u)", "quadBroadcast(1u, -1i)", "quadSwapX(true)",
                  "subgroupElect(true)"])
func subgroupCollectiveOperationsRejectInvalidOverloads(expression: String) {
    let source = "enable subgroups; @compute @workgroup_size(32) fn run() { _ = \(expression); }"
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.subgroup])
    }
}

@Test func signedSubgroupBitwiseReductionsUseUnsignedHLSLWaveOperands() throws {
    let source = ShaderInput.wgsl("""
        enable subgroups;
        @group(0) @binding(0) var<storage, read_write> out: array<vec2i, 32>;
        @compute @workgroup_size(32) fn run(@builtin(local_invocation_index) index: u32) {
            out[index] = subgroupAnd(vec2i(i32(index), -1i));
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .hlsl)
    guard case .text(_, let text) = shader.artifact else { Issue.record("Expected HLSL"); return }
    #expect(text.contains("asint(WaveActiveBitAnd(asuint("))
}

#if canImport(Metal)
@Test func computesSubgroupVotesMasksAndPrefixesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.subgroups])
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(subgroupCollectiveShader), to: .msl))
    let source = ShaderInput.wgsl(subgroupCollectiveFragment)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let output = try await device.createBuffer(.init(size: 14336, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 14336, usage: [.copyDestination, .mapRead]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    let snapshots = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 14336),
        .read(buffer: readback, byteCount: 14336),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)]),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
    ]).wait()
    let words = try #require(snapshots.first)
    try #require(words.count == 3584)
    for index in 0..<128 {
        let base = index * 28
        let size = words[base]
        let lane = words[base + 1]
        try #require(size >= 4 && size <= 128 && size & (size - 1) == 0 && lane < size)
        let ballot = (0..<4).map { word -> UInt32 in
            let remaining = min(32, max(0, Int(size) - word * 32))
            return remaining == 32 ? .max : (UInt32(1) << remaining) - 1
        }
        let expected: [UInt32] = [size, lane, 1, 1] + ballot.map { $0 & 0x55555555 } +
            [size * (size - 1) / 2, 8, 7, size - 1,
             0x80000000, size - 1, 0, lane * (lane + 1) / 2,
             lane * (lane &- 1) / 2, 1 << min(lane, 3), 1 << min(lane + 1, 3), lane == 0 ? 1 : 0] + ballot + [1, 1, 1, 1]
        #expect(Array(words[base..<(base + 28)]) == expected)
    }
    let pixels = try #require(snapshots.last)
    #expect([pixels[0], pixels[1], pixels[64], pixels[65]] == [0x03020101, 0x02030001, 0x01000301, 0x00010201])
    await device.close()
}
#endif
