import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let localUniformityShader = ShaderInput.wgsl("""
    requires subgroup_id, subgroup_uniformity;
    enable subgroups;
    @group(0) @binding(0) var<storage, read_write> output: array<vec2u>;
    fn selected(group: u32) -> bool {
        var flag: bool;
        if group == 0u { flag = true; } else { flag = false; }
        return flag;
    }
    @compute @workgroup_size(128)
    fn run(@builtin(subgroup_id) group: u32, @builtin(local_invocation_index) index: u32) {
        var selector = group;
        let previous = selector;
        selector = index;
        if selected(previous) {
            output[index] = vec2u(group, subgroupAdd(1u));
        } else {
            output[index] = vec2u(group, 0u);
        }
    }
    """)

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func localUniformityRetainsReadSnapshotsAndMergesSubgroupBranches(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(localUniformityShader, to: format)
    if case .text(.wgsl, let text) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.subgroup))
    }
}

@Test(arguments: [
    "var slot = lane; slot = 0u;",
    "var slot = lane; let ptr = &slot; *ptr = 0u;",
    "var slot: u32; if lane == 0u { _ = lane; }",
    "var slot: u32; _ = inspect(&slot);",
    "var slot: u32; loop { break; }",
    "var slot: u32; switch group.x { case 0u: { slot = 1u; break; } default: { slot = 2u; } }",
])
func localUniformityAcceptsUniformValuesReachingABarrier(_ body: String) throws {
    let source = """
        fn inspect(pointer: ptr<function, u32>) -> u32 { return *pointer; }
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            \(body)
            if slot == 0u { workgroupBarrier(); }
        }
        """
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
}

@Test(arguments: [
    "var slot: u32; if lane == 0u { slot = 1u; } if slot == 0u { workgroupBarrier(); }",
    "var slot: u32; switch lane { case 0u: { slot = 1u; break; } default: {} } if slot == 0u { workgroupBarrier(); }",
    "var pair = vec2u(lane); pair.x = 0u; if pair.x == 0u { workgroupBarrier(); }",
    "var slot: u32; replace(&slot, lane); if slot == 0u { workgroupBarrier(); }",
    "var slot: u32; loop { slot = lane; break; } if slot == 0u { workgroupBarrier(); }",
])
func localUniformityRejectsVaryingValuesReachingABarrier(_ body: String) {
    let source = """
        fn replace(pointer: ptr<function, u32>, value: u32) { *pointer = value; }
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            \(body)
        }
        """
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func localUniformitySelectsTheExpectedMetalSubgroupAfterAStore() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(localUniformityShader, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 1024, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 1024, usage: [.mapRead, .copyDestination]))
    let results = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 1024),
        .read(buffer: readback, byteCount: 1024),
    ]).wait()
    let words = try #require(results.first)
    try #require(words.count == 256)
    let indices = stride(from: 0, to: words.count, by: 2)
    let firstGroupSize = UInt32(indices.filter { words[$0] == 0 }.count)
    #expect(firstGroupSize > 0)
    #expect(indices.allSatisfy { words[$0 + 1] == (words[$0] == 0 ? firstGroupSize : 0) })
    await device.close()
}
#endif
