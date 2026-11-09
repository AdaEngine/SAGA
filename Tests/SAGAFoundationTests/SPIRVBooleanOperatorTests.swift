import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvEagerBooleanInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 3>;
    var<private> trace: u32;

    fn scalar(tag: u32, value: bool) -> bool {
        trace = trace * 10u + tag;
        return value;
    }
    fn vector(tag: u32, value: vec4<bool>) -> vec4<bool> {
        trace = trace * 10u + tag;
        return value;
    }
    @compute @workgroup_size(1) fn run() {
        let both = scalar(1u, false) & scalar(2u, true);
        let either = scalar(3u, true) | scalar(4u, false);
        let lhs = vec4(true, true, false, false);
        let rhs = vec4(true, false, true, false);
        let intersection = vector(5u, lhs) & vector(6u, rhs);
        let union = vector(7u, lhs) | vector(8u, rhs);
        let short = (false && scalar(9u, true)) || (true || scalar(9u, false));
        output[0] = vec4u(u32(both), u32(either), trace, u32(short));
        output[1] = select(vec4u(0), vec4u(1), intersection);
        output[2] = select(vec4u(0), vec4u(1), union);
    }
    """)

private func exportEagerBoolean(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvEagerBooleanInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvBooleanOperatorsRoundTripEagerScalarAndVectorOperations(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportEagerBoolean(compiler)), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [48])
    guard case .text(let target, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    if format == .wgsl {
        #expect(text.contains(" & ") && text.contains(" | "))
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvBooleanOperatorsPreserveEagerSideEffectsAndLaneValuesOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let input = try roundTrip ? ShaderInput.spirv(exportEagerBoolean(compiler)) : spirvEagerBooleanInput
    let shader = try compiler.compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
        ]).wait()
        #expect(snapshots == [[0, 1, 12345678, 1, 1, 0, 0, 0, 1, 1, 1, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
