import SAGACompiler
import SAGACore
import SAGAFrontends
import SAGAShaderIR
import SAGATypes
import Testing

#if canImport(Metal)
import SAGAMetal
#endif

private let implicitSubgroupSource = ShaderInput.wgsl("""
    struct Inputs {
        @builtin(num_subgroups) groups: u32,
        @builtin(subgroup_size) size: u32,
    }
    @group(0) @binding(0) var<storage, read_write> output: vec4u;
    @compute @workgroup_size(64)
    fn main(inputs: Inputs, @builtin(subgroup_id) group: u32, @builtin(subgroup_invocation_id) lane: u32) {
        let ballot = subgroupBallot();
        output = vec4u(inputs.groups + group, inputs.size, ballot.x, quadBroadcast(lane, 4u));
    }
    """)

private let narrowSubgroupSource = ShaderInput.wgsl("""
    enable wgpu_int16;
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 256>;
    @compute @workgroup_size(128)
    fn main(@builtin(subgroup_size) size: u32, @builtin(subgroup_invocation_id) lane: u32,
            @builtin(local_invocation_index) index: u32) {
        let signed = i16(lane) - i16(16);
        let factor = select(i16(1), i16(30000), lane < 2u);
        let unsigned = u16(65535) - u16(lane);
        let paired = vec2<i16>(signed, i16(30000));
        let sum = subgroupAdd(paired);
        output[index * 2u] = vec4u(size, lane, u32(bitcast<u16>(sum.y)), u32(bitcast<u16>(subgroupMin(signed))));
        output[index * 2u + 1u] = vec4u(
            u32(bitcast<u16>(subgroupMul(factor))),
            u32(bitcast<u16>(subgroupExclusiveAdd(signed))),
            u32(subgroupMax(unsigned)),
            u32(bitcast<u16>(subgroupBroadcastFirst(subgroupBroadcast(signed, 4u)))),
        );
    }
    """)

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func translatesImplicitNagaSubgroupBuiltinsOnNativeTargets(_ format: ShaderOutputFormat) throws {
    let shader = try IndependentShaderCompiler().compileCompute(implicitSubgroupSource, entryPoint: "main", to: format)
    #expect(shader.requiresSubgroups)
}

@Test func nativeSubgroupSPIRVExportsPortableExplicitBallotAndQuadSelector() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let binary) = try compiler.compile(implicitSubgroupSource, to: .spirv) else {
        Issue.record("Expected SPIR-V")
        return
    }
    let imported = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "main")
    guard case .text(_, let text) = imported.artifact else { Issue.record("Expected WGSL"); return }
    #expect(text.contains("enable subgroups;"))
    #expect(text.contains("requires subgroup_id;"))
    #expect(text.contains("subgroupBallot(true)"))
    #expect(text.contains("quadBroadcast(") && !text.contains(", 4u)"))
    _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.subgroup])
}

@Test func nagaWGSLSourceProfileNormalizesPortableSourceForWebGPU() throws {
    let source = ShaderInput.wgsl("""
        const maximum: f32 = 3.40282347e38;
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(32u, 2i)
        fn main(@builtin(subgroup_invocation_id) lane: u32) {
            let ballot = subgroupBallot();
            output = bitcast<u32>(maximum) ^ quadBroadcast(ballot.x + lane, 4u);
        }
        """)
    let compiler = IndependentShaderCompiler(wgslSourceProfile: .nagaNative)
    let output = try compiler.compileCompute(source, entryPoint: "main", to: .wgsl)
    #expect(output.workgroupSize == .init(x: 32, y: 2))
    #expect(output.requiresSubgroups)
    guard case .text(.wgsl, let text) = output.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(try compiler.compile(source, to: .wgsl) == output.artifact)
    #expect(text.contains("enable subgroups;"))
    #expect(text.contains("subgroupBallot(true)"))
    #expect(text.contains("quadBroadcast(") && !text.contains(", 4u)"))
    let browser = try compiler.compileWebGPUCompute(.wgsl(text))
    #expect(browser.workgroupSize == output.workgroupSize)
    #expect(browser.requiresSubgroups)
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(source) }
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(source, capabilities: [.subgroup]) }
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compile(source, to: .wgsl) }
    let dot = try compiler.compileDOT(source)
    #expect(try compiler.compile(source, to: .dot) == dot)
}

@Test func nagaWGSLSourceProfileEmitsNativeNarrowWGSL() throws {
    let compiler = IndependentShaderCompiler(wgslSourceProfile: .nagaNative)
    let output = try compiler.compileCompute(narrowSubgroupSource, entryPoint: "main", to: .wgsl)
    #expect(output.requiresShaderInt16 && output.requiresSubgroups)
    guard case .text(.wgsl, let text) = output.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("enable wgpu_int16;"))
    let msl = try IndependentShaderCompiler().compileCompute(.wgsl(text), to: .msl)
    #expect(msl.requiresShaderInt16 && msl.requiresSubgroups)
    #expect(throws: ShaderFailure.unsupported("16-bit integer types require native shader output")) {
        try compiler.compileWebGPUCompute(.wgsl(text))
    }
}

@Test func nagaWGSLSourceProfileKeepsBrowserRenderRoundingStrict() throws {
    let source = ShaderInput.wgsl("""
        const maximum: f32 = 3.40282347e38;
        @fragment fn main() -> @location(0) vec4f { return vec4f(maximum); }
        """)
    let compiler = IndependentShaderCompiler(wgslSourceProfile: .nagaNative)
    let output = try compiler.compileRenderStage(source, entryPoint: "main", to: .wgsl)
    guard case .text(.wgsl, let text) = output.artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.compileWebGPURenderStage(.wgsl(text), entryPoint: output.entryPoint)
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPURenderStage(source, entryPoint: "main") }
}

@Test func keepsExplicitWGSLSubgroupRequirementsForWebGPUAndDefaultParser() {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(implicitSubgroupSource, capabilities: [.subgroup])
    }
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(implicitSubgroupSource) }
    #expect(throws: ShaderFailure.self) {
        try WGSLFrontend(options: .init(capabilities: [], allowLegacyNagaSubgroups: true)).parse(implicitSubgroupSource)
    }
    let explicitlyEnabled = ShaderInput.wgsl("""
        enable subgroups;
        @compute @workgroup_size(64) fn main() { _ = quadBroadcast(1u, 4u); }
        """)
    #expect(throws: ShaderFailure.self) { try compiler.compileCompute(explicitlyEnabled, to: .msl) }
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(explicitlyEnabled) }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func translatesNarrowNagaSubgroupReductionsAndGathers(_ format: ShaderOutputFormat) throws {
    let shader = try IndependentShaderCompiler().compileCompute(narrowSubgroupSource, entryPoint: "main", to: format)
    #expect(shader.requiresSubgroups && shader.requiresShaderInt16)
}

@Test func narrowNagaSubgroupSpirvRoundTripPreservesNativeProfile() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let binary) = try compiler.compile(narrowSubgroupSource, to: .spirv) else {
        Issue.record("Expected SPIR-V")
        return
    }
    let imported = try compiler.compileCompute(.spirv(binary), entryPoint: "main", to: .msl)
    #expect(imported.requiresSubgroups && imported.requiresShaderInt16)
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "main") }
}

#if canImport(Metal)
@Test func nativeNarrowSubgroupLiftingPreservesExactMetalBits() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(narrowSubgroupSource, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.subgroups, .shaderInt16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let bytes: UInt64 = 4096
        let output = try await device.createBuffer(.init(size: bytes, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: bytes, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: bytes), .read(buffer: readback, byteCount: bytes),
        ]).wait()
        let words = try #require(snapshots.first)
        try #require(words.count == 1024)
        for base in stride(from: 0, to: words.count, by: 8) {
            let size = Int(words[base]), lane = Int(words[base + 1])
            try #require((8...128).contains(size) && 128 % size == 0 && lane < size)
            let prefix = lane * (lane - 1) / 2 - 16 * lane
            let expected: [UInt32] = [
                UInt32(size), UInt32(lane), UInt32(UInt16(truncatingIfNeeded: size * 30_000)), 65_520,
                UInt32(UInt16(truncatingIfNeeded: 30_000 * 30_000)),
                UInt32(UInt16(truncatingIfNeeded: prefix)), 65_535, 65_524,
            ]
            #expect(Array(words[base..<(base + 8)]) == expected)
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
