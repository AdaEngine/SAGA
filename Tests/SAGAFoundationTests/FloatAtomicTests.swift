import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let floatAtomicShader = """
    struct Data { counters: array<array<atomic<f32>, 2>, 1>, @size(12) sentinel: u32 }
    @group(0) @binding(0) var<storage, read_write> scalar: atomic<f32>;
    @group(0) @binding(1) var<storage, read_write> data: Data;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 4>;
    var<workgroup> sharedValue: atomic<f32>;

    fn increment(p: ptr<storage, atomic<f32>, read_write>) -> f32 {
        return atomicAdd(p, 0.5);
    }

    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32) {
        if lane == 0u {
            atomicStore(&scalar, -0.0);
            atomicStore(&data.counters[0][0], 1.25);
            atomicStore(&data.counters[0][1], 5.0);
            atomicStore(&sharedValue, -0.0);
        }
        storageBarrier();
        workgroupBarrier();
        let sharedBits = bitcast<u32>(workgroupUniformLoad(&sharedValue));
        let before = increment(&scalar);
        let nested = atomicSub(&data.counters[0][0], 0.25);
        let invalid = increment(&data.counters[0][lane + 2u]);
        output[lane] = vec4u(bitcast<u32>(before), bitcast<u32>(nested), bitcast<u32>(invalid), sharedBits);
        storageBarrier();
        if lane == 0u {
            output[0].z = bitcast<u32>(atomicExchange(&data.counters[0][1], -0.0));
            output[0].w = bitcast<u32>(atomicLoad(&scalar));
            output[1].w = bitcast<u32>(atomicLoad(&data.counters[0][0]));
            output[2].w = bitcast<u32>(atomicLoad(&data.counters[0][1]));
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func floatAtomicOperationsCompileAcrossTargets(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(floatAtomicShader)
    let shader = try compiler.compileCompute(source, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [4, 20, 64])
    guard case .text(let actual, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl {
        #expect(code.contains("bitcast<f32>(0x80000000u)"))
        _ = try compiler.parseAndValidate(.wgsl(code))
    }
}

@Test(arguments: [
    "atomicMin(&value, 1.0);", "_ = atomicCompareExchangeWeak(&value, 0.0, 1.0);",
    "_ = atomicAdd(&sharedValue, 1.0);",
])
func floatAtomicOperationsEnforceNagaOperatorAndScopeLimits(_ statement: String) throws {
    let input = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> value: atomic<f32>;
        var<workgroup> sharedValue: atomic<f32>;
        @compute @workgroup_size(1) fn run() { \(statement) }
        """)
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input, capabilities: [.shaderFloat32Atomic])
        Issue.record("Accepted an unsupported float atomic operation")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == (statement.contains("sharedValue") ? .invalidAddressSpace : .typeMismatch))
    }
}

@Test func floatAtomicTypesRequireTheirCapabilityForValidation() {
    let input = ShaderInput.wgsl("@group(0) @binding(0) var<storage, read_write> value: atomic<f32>;")
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}

#if canImport(Metal)
@Test func floatAtomicOperationsReturnPreviousBitsUnderContentionOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(floatAtomicShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let scalar = try await device.createBuffer(.init(size: 4, usage: .storage))
        let data = try await device.createBuffer(.init(size: 20, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .write(buffer: data, offset: 8, words: [99, 100, 101]),
            .dispatchCompute(pipeline: pipeline, bindings: [
                .init(binding: 0): scalar, .init(binding: 1): data, .init(binding: 2): output,
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
            .copy(source: data, destination: readback, byteCount: 20), .read(buffer: readback, byteCount: 20),
        ]).wait()
        try #require(snapshots.count == 2 && snapshots[0].count == 16)
        let result = snapshots[0]
        #expect([0, 4, 8, 12].map { result[$0] }.sorted() == [0x3f000000, 0x3f800000, 0x3fc00000, 0x80000000])
        #expect([1, 5, 9, 13].map { result[$0] }.sorted() == [0x3f000000, 0x3f400000, 0x3f800000, 0x3fa00000])
        #expect([2, 6, 10, 14].map { result[$0] } == [0x40a00000, 0, 0, 0])
        #expect([3, 7, 11, 15].map { result[$0] } == [0x40000000, 0x3e800000, 0x80000000, 0x80000000])
        #expect(snapshots[1] == [0x3e800000, 0x80000000, 99, 100, 101])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
