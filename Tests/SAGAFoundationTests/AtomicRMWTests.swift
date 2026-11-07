import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let atomicRMWShader = """
    @group(0) @binding(0) var<storage, read_write> signedValue: atomic<i32>;
    @group(0) @binding(1) var<storage, read_write> unsignedValues: array<atomic<u32>>;
    @group(0) @binding(2) var<storage, read_write> output: array<u32, 17>;
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        output[0] = u32(atomicSub(&signedValue, i32(2147483648u)));
        output[1] = u32(atomicMin(&signedValue, -7i));
        output[2] = u32(atomicMax(&signedValue, -7i));
        output[3] = u32(atomicAnd(&signedValue, 15i));
        output[4] = u32(atomicOr(&signedValue, 32i));
        output[5] = u32(atomicXor(&signedValue, 3i));
        output[6] = u32(atomicExchange(&signedValue, -1i));
        atomicOr(&signedValue, 0i);
        output[7] = u32(atomicLoad(&signedValue));
        output[8] = atomicSub(&unsignedValues[0], 1u);
        output[9] = atomicMax(&unsignedValues[0], 7u);
        output[10] = atomicMin(&unsignedValues[0], 7u);
        output[11] = atomicAnd(&unsignedValues[0], 6u);
        output[12] = atomicOr(&unsignedValues[0], 8u);
        output[13] = atomicXor(&unsignedValues[0], 3u);
        output[14] = atomicExchange(&unsignedValues[0], 23u);
        output[15] = atomicLoad(&unsignedValues[0]);
        output[16] = atomicExchange(&unsignedValues[id.x + 1u], 777u);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsSignedAndUnsignedAtomicRMW(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(atomicRMWShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    let expected: [String]
    switch format {
    case .msl:
        expected = ["atomic_fetch_sub_explicit", "atomic_fetch_min_explicit", "atomic_fetch_max_explicit",
                    "atomic_fetch_and_explicit", "atomic_fetch_or_explicit", "atomic_fetch_xor_explicit",
                    "atomic_exchange_explicit", "device atomic_int*"]
    case .hlsl:
        expected = ["InterlockedAdd", "InterlockedMin", "InterlockedMax", "InterlockedAnd", "InterlockedOr",
                    "InterlockedXor", "InterlockedExchange", "RWStructuredBuffer<int>", "asint(0u -"]
    case .glsl:
        expected = ["atomicAdd", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor", "atomicExchange",
                    "int(0u -"]
    case .wgsl:
        expected = ["atomicSub", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor", "atomicExchange"]
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target"); return
    }
    for spelling in expected { #expect(source.contains(spelling)) }
}

#if canImport(Metal)
@Test func atomicRMWReturnsOldValuesAndWrapsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(atomicRMWShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let signed = try await device.createBuffer(.init(size: 4, usage: .storage))
        let unsigned = try await device.createBuffer(.init(size: 8, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 68, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 68, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: unsigned, offset: 4, words: [999]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0): .init(buffer: signed),
                .init(binding: 1): .init(buffer: unsigned, size: 4),
                .init(binding: 2): .init(buffer: output),
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 68),
            .read(buffer: readback, byteCount: 68),
            .copy(source: unsigned, destination: readback, byteCount: 8),
            .read(buffer: readback, byteCount: 8),
        ])
        let expected: [UInt32] = [
            0, 0x80000000, 0x80000000, 0xfffffff9, 9, 41, 42, .max,
            0, .max, .max, 7, 6, 14, 13, 23, 0,
        ]
        #expect(try await ticket.wait() == [expected, [23, 999]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
