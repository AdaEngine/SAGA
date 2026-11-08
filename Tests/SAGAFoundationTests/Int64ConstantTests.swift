import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let int64ConstantShader = """
    const maximum: u64 = 18446744073709551615lu;
    const minimum: i64 = -9223372036854775808li;
    const high_word: u64 = maximum >> 32u;
    const wrapped: u64 = maximum + 3lu;
    const truncated: u32 = u32(maximum);
    const ordered: bool = maximum > 0x8000000000000000lu;
    const bounded: u64 = clamp(maximum, 0x8000000000000000lu, maximum);
    const_assert u64(18446744073709551616.0f) == 18446742974197923840lu;
    const_assert i64(9223372036854775808.0f) == 9223371487098961920li;
    const lanes = bitcast<vec2<i64>>(vec2<u64>(maximum, 0x8000000000000000lu));
    var<private> state: i64 = minimum;
    @group(0) @binding(0) var<storage, read_write> output: array<u64, 5>;
    @compute @workgroup_size(1) fn main() {
        state += i64(wrapped);
        output[0] = bitcast<u64>(state);
        output[1] = select(0lu, high_word, ordered);
        output[2] = u64(truncated);
        output[3] = min(abs(bitcast<vec2<u64>>(lanes).x), bounded);
        let small = i64(i32(state));
        let product = dot(vec2<i64>(small), vec2<i64>(3li));
        output[4] = u64(clamp(product, -3li, 9li) + sign(small) + abs(-3li));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func int64ConstantsAndConversionsCompile(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(int64ConstantShader), to: format)
    #expect(shader.requiresShaderInt64)
    if case .text(.wgsl, let text) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.shaderInt64))
    }
}

@Test(arguments: [
    "const value = u64(-1);",
    "const value = 1lu << 64u;",
    "const value = 1lu / 0lu;",
    "const value = -9223372036854775808li / -1li;",
    "const value = 1lu + 1li;",
    "@group(1lu) @binding(0) var<storage> value: u32;",
    "@compute @workgroup_size(1) fn main() { switch 1 { case 1li: {} default: {} } }",
])
func int64ConstantsRejectInvalidValues(_ declaration: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(declaration), capabilities: .referenceDefault.union(.shaderInt64))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func int64FixedArrayMembersCopyAsValues(format: ShaderOutputFormat) throws {
    let source = """
        struct Values { signed: array<i64, 2>, unsigned: array<vec2<u64>, 2>, }
        @group(0) @binding(0) var<storage> input: Values;
        @group(0) @binding(1) var<storage, read_write> output: Values;
        @compute @workgroup_size(1) fn main() {
            output.signed = input.signed;
            output.unsigned = input.unsigned;
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [48, 48])
    if case .text(.wgsl, let text) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.shaderInt64))
    }
}

#if canImport(Metal)
@Test func int64ConstantsAndConversionsPreserveBitsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(int64ConstantShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 40, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 40, usage: [.mapRead, .copyDestination]))
    let words = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 40),
        .read(buffer: readback, byteCount: 40),
    ]).wait()
    #expect(words == [[2, 0x8000_0000, .max, 0, .max, 0, .max, .max, 13, 0]])
    await device.close()
}
#endif
