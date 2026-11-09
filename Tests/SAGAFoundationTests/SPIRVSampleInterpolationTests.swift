import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal
#endif

private let spirvSampleInterpolationInput = ShaderInput.wgsl("""
    struct Varyings {
        @location(0) @interpolate(perspective, sample) perspective: f32,
        @location(1) @interpolate(linear, sample) affine: f32,
    }
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(perspective, sample) perspective: f32,
        @location(1) @interpolate(linear, sample) affine: f32,
    }
    @vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOutput {
        if index == 0u { return VertexOutput(vec4f(-1, -1, 0, 1), 0.0, 0.0); }
        if index == 1u { return VertexOutput(vec4f(6, -2, 0, 2), 1.0, 1.0); }
        return VertexOutput(vec4f(-4, 12, 0, 4), 0.0, 0.0);
    }
    @fragment fn fs(input: Varyings) -> @location(0) vec4u {
        return vec4u(u32(round(128.0 * input.affine)), u32(round(128.0 * input.perspective)), 0u, 255u);
    }
    """)

private func exportSampleInterpolation(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvSampleInterpolationInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvSampleInterpolationDeclaresSampleRateShadingAndVaryingDecorations() throws {
    let words = try exportSampleInterpolation(IndependentShaderCompiler())
    var capabilities: Set<UInt32> = []
    var builtins: Set<UInt32> = []
    var sample: Set<UInt32> = []
    var linear: Set<UInt32> = []
    var locations: [UInt32: UInt32] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 71 where count == 4 && words[offset + 2] == 11: builtins.insert(words[offset + 3])
        case 71 where count == 3 && words[offset + 2] == 13: linear.insert(words[offset + 1])
        case 71 where count == 3 && words[offset + 2] == 17: sample.insert(words[offset + 1])
        case 71 where count == 4 && words[offset + 2] == 30: locations[words[offset + 1]] = words[offset + 3]
        default: break
        }
        offset += count
    }
    #expect(capabilities.contains(35)) // SampleRateShading
    #expect(!builtins.contains(18)) // SampleId is unnecessary when interpolation itself is per sample.
    #expect(sample.compactMap { locations[$0] }.sorted() == [0, 0, 1, 1])
    #expect(linear.compactMap { locations[$0] }.sorted() == [1, 1])
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvSampleInterpolationRoundTripsLinkedVaryings(_ format: ShaderOutputFormat, _ roundTrip: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = try roundTrip ? ShaderInput.spirv(exportSampleInterpolation(compiler)) : spirvSampleInterpolationInput
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: format)
    #expect(vertex.varyings == fragment.varyings)
    #expect(fragment.varyings.map(\.sampling) == [.sample, .sample])
    #expect(fragment.varyings.map(\.interpolation) == [.perspective, .linear])
    guard case .text(_, let text) = fragment.artifact else { throw ShaderFailure.incompatibleArtifact }
    switch format {
    case .wgsl:
        #expect(text.contains("@interpolate(perspective, sample)") && text.contains("@interpolate(linear, sample)"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.linearInterpolation))
    case .msl: #expect(text.contains("sample_perspective") && text.contains("sample_no_perspective"))
    case .hlsl: #expect(text.contains("sample float") && text.contains("noperspective sample float"))
    case .glsl: #expect(text.contains("sample in float") && text.contains("noperspective sample in float"))
    default: Issue.record("Unexpected text target")
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvSampleInterpolationUsesActualMetalSamplePositions(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let input = try roundTrip ? ShaderInput.spirv(exportSampleInterpolation(compiler)) : spirvSampleInterpolationInput
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .msl)
    let reader = try compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var color: texture_multisampled_2d<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            output[lane] = textureLoad(color, vec2i(0), lane);
        }
        """), to: .msl)
    let native = try #require(MTLCreateSystemDefaultDevice())
    let positions = native.getDefaultSamplePositions(sampleCount: 4)
    let expected = positions.flatMap { sample -> [UInt32] in
        // Clip-space w is 1, 2, 4 and the varying is 1 only at the second vertex.
        let b = Double(sample.x) / 2
        let c = (1 - Double(sample.y)) / 2
        let a = 1 - b - c
        let perspective = (b / 2) / (a + b / 2 + c / 4)
        return [UInt32((128 * b).rounded(.toNearestOrEven)), UInt32((128 * perspective).rounded(.toNearestOrEven)), 0, 255]
    }
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment,
                                                            colorFormat: .rgba8Uint, sampleCount: 4)
        let compute = try await device.createComputePipeline(reader)
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
            usage: [.renderAttachment, .textureBinding], sampleCount: 4))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3),
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 1): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): color]),
            .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        ]).wait()
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
