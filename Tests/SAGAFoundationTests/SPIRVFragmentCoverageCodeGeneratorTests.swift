import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvFragmentCoverageInput = ShaderInput.wgsl("""
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) marker: f32,
    }
    struct SampleInput {
        @builtin(sample_mask) coverage: u32,
        @location(0) marker: f32,
        @builtin(sample_index) index: u32,
    }
    struct SampleOutput {
        @builtin(frag_depth) depth: f32,
        @builtin(sample_mask) coverage: u32,
        @location(0) color: vec4u,
    }

    @vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOutput {
        if index == 0u { return VertexOutput(vec4f(-1, -1, 0.75, 1), 1.0); }
        if index == 1u { return VertexOutput(vec4f(3, -1, 0.75, 1), 1.0); }
        return VertexOutput(vec4f(-1, 3, 0.75, 1), 1.0);
    }

    @fragment fn fs(input: SampleInput) -> SampleOutput {
        let covered = (input.coverage & (1u << input.index)) != 0u && input.marker == 1.0;
        let depth = f32(input.index + 2u) * 0.0625;
        return SampleOutput(depth, 5u, vec4u(input.index + 1u, select(0u, 9u, covered), 0u, 1u));
    }
    """)

private func exportFragmentCoverage(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvFragmentCoverageInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvFragmentCoverageCodeGeneratorDeclaresSampleAndDepthInterfaces() throws {
    let words = try exportFragmentCoverage(IndependentShaderCompiler())
    var capabilities: Set<UInt32> = []
    var depthReplacing: Set<UInt32> = []
    var fragmentEntry: UInt32?
    var fragmentInterface: Set<UInt32> = []
    var builtins: [UInt32: UInt32] = [:]
    var pointerTargets: [UInt32: UInt32] = [:]
    var variables: [UInt32: (pointer: UInt32, storage: UInt32)] = [:]
    var arrays: [UInt32: (element: UInt32, length: UInt32)] = [:]
    var integers32: Set<UInt32> = []
    var floats32: Set<UInt32> = []
    var constants: [UInt32: UInt32] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 15 where count >= 4 && words[offset + 1] == 4:
            #expect(words[offset + 3] == 0x00007366) // "fs" and NUL in one word.
            fragmentEntry = words[offset + 2]
            fragmentInterface = Set(words[(offset + 4)..<(offset + count)])
        case 16 where count == 3 && words[offset + 2] == 12: depthReplacing.insert(words[offset + 1])
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 21 where count == 4 && words[offset + 2] == 32: integers32.insert(words[offset + 1])
        case 22 where count == 3 && words[offset + 2] == 32: floats32.insert(words[offset + 1])
        case 28 where count == 4: arrays[words[offset + 1]] = (words[offset + 2], words[offset + 3])
        case 32 where count == 4: pointerTargets[words[offset + 1]] = words[offset + 3]
        case 43 where count == 4: constants[words[offset + 2]] = words[offset + 3]
        case 59 where count >= 4: variables[words[offset + 2]] = (words[offset + 1], words[offset + 3])
        case 71 where count == 4 && words[offset + 2] == 11: builtins[words[offset + 1]] = words[offset + 3]
        default: break
        }
        offset += count
    }
    #expect(capabilities.contains(35)) // SampleRateShading
    let entry = try #require(fragmentEntry)
    #expect(depthReplacing == [entry])
    let sampleID = try #require(builtins.first(where: { $0.value == 18 })?.key)
    let sample = try #require(variables[sampleID])
    let sampleType = try #require(pointerTargets[sample.pointer])
    #expect(fragmentInterface.contains(sampleID) && sample.storage == 1 && integers32.contains(sampleType))
    let masks = builtins.filter { $0.value == 20 }.keys
    try #require(masks.count == 2)
    var maskSpaces: Set<UInt32> = []
    for id in masks {
        let variable = try #require(variables[id])
        let target = try #require(pointerTargets[variable.pointer])
        let mask = try #require(arrays[target])
        #expect(fragmentInterface.contains(id) && constants[mask.length] == 1 && integers32.contains(mask.element))
        maskSpaces.insert(variable.storage)
    }
    #expect(maskSpaces == [1, 3]) // Input and Output.
    let depthID = try #require(builtins.first(where: { $0.value == 22 })?.key)
    let depth = try #require(variables[depthID])
    let depthType = try #require(pointerTargets[depth.pointer])
    #expect(fragmentInterface.contains(depthID) && depth.storage == 3 && floats32.contains(depthType))
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvFragmentCoverageCodeGeneratorRoundTripsSampleMaskAndDepth(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(try exportFragmentCoverage(compiler))
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: format)
    #expect(vertex.varyings == fragment.varyings)
    #expect(fragment.writesDepth && fragment.writesSampleMask && fragment.colorOutputKind == .uint)
    guard case .text(_, let text) = fragment.artifact else { throw ShaderFailure.incompatibleArtifact }
    switch format {
    case .wgsl:
        #expect(text.contains("@builtin(sample_index)") && text.contains("@builtin(sample_mask)"))
        #expect(text.contains("@builtin(frag_depth)"))
        _ = try compiler.parseAndValidate(.wgsl(text))
    case .msl:
        #expect(text.contains("[[sample_id]]") && text.contains("[[sample_mask]]") && text.contains("[[depth(any)]]"))
    case .hlsl:
        #expect(text.contains("SV_SampleIndex") && text.contains("SV_Coverage") && text.contains("SV_Depth"))
    case .glsl:
        #expect(text.contains("gl_SampleID") && text.contains("gl_SampleMaskIn[0]") && text.contains("gl_SampleMask[0]"))
        #expect(text.contains("gl_FragDepth"))
    default: Issue.record("Unexpected text target")
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvFragmentCoverageCodeGeneratorPreservesIndividualColorAndDepthSamplesOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let input = try roundTrip ? ShaderInput.spirv(exportFragmentCoverage(compiler)) : spirvFragmentCoverageInput
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .msl)
    let reader = try compiler.compileCompute(.wgsl("""
        struct Readback { color: array<vec4u, 4>, depth: array<f32, 4> }
        @group(0) @binding(0) var color: texture_multisampled_2d<u32>;
        @group(0) @binding(1) var depth: texture_depth_multisampled_2d;
        @group(0) @binding(2) var<storage, read_write> output: Readback;
        @compute @workgroup_size(4) fn readSamples(@builtin(local_invocation_index) lane: u32) {
            output.color[lane] = textureLoad(color, vec2i(0), lane);
            output.depth[lane] = textureLoad(depth, vec2i(0), lane);
        }
        """), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
            depthState: .init(compare: .less), sampleCount: 4, sampleMask: 0b0110)
        let compute = try await device.createComputePipeline(reader)
        let usage: TextureUsage = [.renderAttachment, .textureBinding]
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: usage, sampleCount: 4))
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float, usage: usage, sampleCount: 4))
        let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 80, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .clear(0.5))),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): color, .init(binding: 1): depth]),
            .copy(source: output, destination: readback, byteCount: 80), .read(buffer: readback, byteCount: 80),
        ]).wait()
        let expected: [UInt32] = [0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0, 1, 0, 0, 0, 0] +
            [Float(0.5), 0.5, 0.25, 0.5].map(\.bitPattern)
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
