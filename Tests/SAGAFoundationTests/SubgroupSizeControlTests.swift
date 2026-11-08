import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let subgroupSizeShader = ShaderInput.wgsl("""
    enable subgroup_size_control;
    @id(7) override size: u32 = 32u;
    override width: u32 = 128u;
    @group(0) @binding(0) var<storage, read_write> output: array<vec2u>;
    @compute @workgroup_size(width) @subgroup_size(size * 1u)
    fn run(@builtin(local_invocation_index) index: u32,
           @builtin(subgroup_size) width: u32, @builtin(subgroup_invocation_id) lane: u32) {
        output[index] = vec2u(width, lane);
    }
    """)

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func specializesSubgroupSizeInComputeShaders(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(subgroupSizeShader, to: format, constants: ["7": 16, "width": 64])
    #expect(shader.workgroupSize.x == 64)
    #expect(shader.requiredSubgroupSize == 16)
    guard case .text(_, let text) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(text.contains("enable subgroup_size_control;"))
        #expect(text.contains("@subgroup_size(16)"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.subgroup))
    case .msl:
        #expect(text.contains("threads_per_simdgroup"))
    case .hlsl:
        #expect(shader.hlslProfile == "cs_6_6")
        #expect(text.contains("[WaveSize(16)]"))
        #expect(text.contains("[numthreads(64, 1, 1)]"))
    case .glsl:
        #expect(text.contains("gl_SubgroupSize"))
    default: Issue.record("Unexpected shader target")
    }
}

@Test(arguments: [
    "@compute @workgroup_size(32) @subgroup_size(32) fn run() {}",
    "enable subgroups; @compute @workgroup_size(32) @subgroup_size(32) fn run() {}",
    "enable subgroup_size_control; @compute @workgroup_size(32) @subgroup_size(0) fn run() {}",
    "enable subgroup_size_control; @compute @workgroup_size(32) @subgroup_size(3) fn run() {}",
    "enable subgroup_size_control; @compute @workgroup_size(32) @subgroup_size(16lu) fn run() {}",
    "enable subgroup_size_control; @compute @workgroup_size(32) @subgroup_size(16.0f) fn run() {}",
    "enable subgroup_size_control; @fragment @subgroup_size(16) fn run() {}",
    "enable subgroup_size_control; @subgroup_size(16) fn helper() {}",
])
func rejectsInvalidSubgroupSizeDeclarations(_ source: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.subgroup))
    }
}

@Test func validatesSubgroupSizeAfterPipelineSpecialization() throws {
    let compiler = IndependentShaderCompiler()
    for constants: [String: Double] in [["7": 24], ["7": 32, "width": 16], ["7": 0]] {
        #expect(throws: ShaderFailure.self) {
            try compiler.compileCompute(subgroupSizeShader, to: .wgsl, constants: constants)
        }
    }
    let invalidX = ShaderInput.wgsl("""
        enable subgroup_size_control;
        @compute @workgroup_size(16, 2) @subgroup_size(32) fn run() {}
        """)
    _ = try compiler.parseAndValidate(invalidX, capabilities: .referenceDefault.union(.subgroup))
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(invalidX, to: .msl)
    }
}

@Test func subgroupSizeControlChecksBackendLimits() throws {
    let available = try DeviceCapabilities(maxBufferSize: 1024, features: [.subgroups, .subgroupSizeControl],
                                           subgroupSizeRange: 32...64)
    let unavailable = try DeviceCapabilities(maxBufferSize: 1024, features: [.subgroups])
    try available.requireSubgroupSize(32)
    try available.requireSubgroupSize(64)
    #expect(throws: GPUError.validation("Requested subgroup size is outside the device range")) {
        try available.requireSubgroupSize(16)
    }
    #expect(throws: GPUError.validation("Feature not advertised")) {
        try unavailable.requireSubgroupSize(32)
    }
}

#if canImport(Metal)
@Test func subgroupSizeControlMatchesTheNativeMetalWidth() async throws {
    let compiler = IndependentShaderCompiler()
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let probe = try compiler.compileCompute(.wgsl("""
        enable subgroups;
        @group(0) @binding(0) var<storage, read_write> output: array<vec2u>;
        @compute @workgroup_size(128)
        fn run(@builtin(local_invocation_index) index: u32,
               @builtin(subgroup_size) width: u32, @builtin(subgroup_invocation_id) lane: u32) {
            output[index] = vec2u(width, lane);
        }
        """), to: .msl)
    let output = try await device.createBuffer(.init(size: 1024, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 1024, usage: [.mapRead, .copyDestination]))
    func execute(_ pipeline: ComputePipeline) async throws -> [UInt32] {
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 1024),
            .read(buffer: readback, byteCount: 1024),
        ]).wait()
        return try #require(snapshots.first)
    }
    let ordinary = try await execute(device.createComputePipeline(probe))
    let nativeWidth = try #require(ordinary.first)
    try #require(nativeWidth > 0)
    let fixed = try compiler.compileCompute(subgroupSizeShader, to: .msl, constants: ["7": Double(nativeWidth)])
    let fixedPipeline = try await device.createComputePipeline(fixed)
    #expect(try await execute(fixedPipeline) == ordinary)
    let incompatible = try compiler.compileCompute(
        subgroupSizeShader, to: .msl, constants: ["7": nativeWidth == 32 ? 64 : 32])
    await #expect(throws: GPUError.validation("Requested subgroup size does not match the Metal pipeline SIMD width")) {
        try await device.createComputePipeline(incompatible)
    }
    await device.close()
}
#endif
