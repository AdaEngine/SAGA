import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes

private let int64AtomicBindings = """
    struct Counters {
        signed_value: atomic<i64>,
        unsigned_values: array<atomic<u64>>,
    }
    @group(0) @binding(0) var<storage, read_write> counters: Counters;
    @group(0) @binding(1) var<storage, read_write> flat: atomic<i64>;
    @group(0) @binding(2) var<storage, read_write> result: array<u32>;
    """

private let int64AtomicStorageOps = """
    atomicStore(&counters.signed_value, -9223372036854775808li);
    let signed_load = atomicLoad(&counters.signed_value);
    let old_min = atomicMin(&counters.signed_value, -1li);
    let old_sub = atomicSub(&counters.signed_value, signed_load);
    let old_xor = atomicXor(&counters.signed_value, -1li);
    let swap_signed = atomicCompareExchangeWeak(&counters.signed_value, -1li, -2li);
    atomicStore(&flat, swap_signed.old_value);
    let old_and = atomicAnd(&flat, -2li);
    let old_max = atomicMax(&counters.unsigned_values[lane], 0x8000000000000000lu);
    let old_add = atomicAdd(&counters.unsigned_values[lane], 7lu);
    let old_exchange = atomicExchange(&counters.unsigned_values[lane], old_max);
    let swap_unsigned = atomicCompareExchangeWeak(&counters.unsigned_values[lane], old_exchange, old_add);
    if swap_unsigned.exchanged && swap_signed.exchanged && old_min == old_sub && old_xor == old_and {
        result[lane] = 1u;
    }
    """

private let int64AtomicAllOpsSource = """
    \(int64AtomicBindings)
    struct GroupValues {
        signed_value: atomic<i64>,
        unsigned_values: array<atomic<u64>, 2>,
    }
    var<workgroup> group_values: GroupValues;
    var<workgroup> lanes: array<atomic<u64>, 2>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
        let zero = atomicLoad(&lanes[lane]);
        atomicStore(&group_values.unsigned_values[lane], 0x8000000000000000lu);
        let old_or = atomicOr(&group_values.signed_value, -4li);
        let group_and = atomicAnd(&group_values.signed_value, -2li);
        let group_xor = atomicXor(&group_values.signed_value, -8li);
        let group_max = atomicMax(&lanes[lane], zero);
        let shared_swap = atomicCompareExchangeWeak(&group_values.signed_value, group_xor, group_and);
        if shared_swap.exchanged && shared_swap.old_value == old_or && group_max == 0lu {
            result[lane] = 2u;
        }
        \(int64AtomicStorageOps)
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .hlsl, .glsl], [false, true])
func bufferInt64AtomicAllOpsCompilesStorageAndWorkgroup(format: ShaderOutputFormat, roundTrip: Bool) throws {
    let compiler = IndependentShaderCompiler()
    var input = ShaderInput.wgsl(int64AtomicAllOpsSource)
    if roundTrip {
        guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
            throw ShaderFailure.incompatibleArtifact
        }
        input = .spirv(words)
    }
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.workgroupStorageByteSize == 48)
    #expect(shader.requiresShaderInt64AtomicAllOps)
    #expect(!shader.requiresShaderInt64AtomicMinMax)
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    switch format {
    case .wgsl:
        #expect(source.contains("atomicCompareExchangeWeak("))
        #expect(source.contains("saga_member_0: i64"))
        _ = try compiler.parseAndValidate(.wgsl(source),
            capabilities: .referenceDefault.union([.shaderInt64, .shaderInt64AtomicAllOps]))
    case .hlsl:
        #expect(shader.hlslProfile == "cs_6_6")
        #expect(source.contains("InterlockedExchange64("))
        #expect(source.contains("InterlockedCompareExchange64("))
        #expect(source.contains("InterlockedXor64("))
        #expect(source.contains("groupshared"))
    case .glsl:
        #expect(source.contains("GL_EXT_shader_atomic_int64"))
        #expect(source.contains("atomicCompSwap("))
        #expect(source.contains("shared"))
        #expect(source.contains("atomicExchange("))
    default: Issue.record("Unexpected output format")
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .hlsl, .glsl])
func bufferInt64AtomicAllOpsCompilesFragmentComparison(format: ShaderOutputFormat) throws {
    let source = """
        \(int64AtomicBindings)
        @fragment fn run(@builtin(position) pixel: vec4f) {
            let lane = u32(pixel.x);
            \(int64AtomicStorageOps)
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(source), entryPoint: "run", to: format)
    #expect(shader.requiresShaderInt64AtomicAllOps)
    #expect(!shader.requiresShaderInt64AtomicMinMax)
    guard case .text(_, let text) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    #expect(text.contains("struct SagaAtomicResultI64"))
    #expect(text.contains("struct SagaAtomicResultU64"))
    if format == .hlsl { #expect(shader.hlslProfile == "ps_6_6") }
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(text),
            capabilities: .referenceDefault.union([.shaderInt64, .shaderInt64AtomicAllOps]))
    }
}

@Test func bufferInt64AtomicAllOpsRequiresNativeFullTier() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(int64AtomicAllOpsSource)
    _ = try compiler.parseAndValidate(input, capabilities: .referenceDefault.union([.shaderInt64, .shaderInt64AtomicAllOps]))
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(input, capabilities: .referenceDefault.union([.shaderInt64, .shaderInt64AtomicMinMax]))
    }
    #expect(throws: ShaderFailure.self) { try compiler.compileCompute(input, to: .msl) }
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(input) }
    let minMax = try DeviceCapabilities(maxBufferSize: 256, features: [.shaderInt64AtomicMinMax])
    let allOps = try DeviceCapabilities(maxBufferSize: 256, features: [.shaderInt64AtomicAllOps])
    #expect(throws: GPUError.self) { try minMax.requireShaderInt64Atomics(minMax: false, allOps: true) }
    try allOps.requireShaderInt64Atomics(minMax: true, allOps: false)
}
