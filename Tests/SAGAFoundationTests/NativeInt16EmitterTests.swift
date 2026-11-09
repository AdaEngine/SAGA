import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing

#if canImport(Metal)
import SAGAMetal
#endif

private let narrowIntegerFloatSource = ShaderInput.wgsl("""
    enable wgpu_int16;
    @group(0) @binding(0) var<storage, read_write> words: array<u32, 8>;
    @compute @workgroup_size(1) fn main() {
        let s = i16(bitcast<i32>(words[0]));
        let u = u16(words[1]);
        let sv = vec2f(vec2<i16>(s, i16(-13)));
        let uv = vec2f(vec2<u16>(u, u16(50001)));
        words[2] = bitcast<u32>(f32(s));
        words[3] = bitcast<u32>(f32(u));
        words[4] = bitcast<u32>(sv.x);
        words[5] = bitcast<u32>(sv.y);
        words[6] = bitcast<u32>(uv.x);
        words[7] = bitcast<u32>(uv.y);
    }
    """)

private func narrowIntegerFloatSPIRV(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let binary) = try compiler.compile(narrowIntegerFloatSource, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return binary
}

@Test func spirvInt16ToFloatConversionsPreserveScalarAndVectorSignedness() throws {
    let compiler = IndependentShaderCompiler()
    let binary = try narrowIntegerFloatSPIRV(compiler)
    var opcodes: [UInt32: Int] = [:]
    var offset = 5
    while offset < binary.count {
        let count = Int(binary[offset] >> 16)
        try #require(count > 0 && count <= binary.count - offset)
        opcodes[binary[offset] & 0xffff, default: 0] += 1
        offset += count
    }
    #expect(opcodes[111] == 2) // OpConvertSToF
    #expect(opcodes[112] == 2) // OpConvertUToF
    let imported = try compiler.compileCompute(.spirv(binary), entryPoint: "main", to: .msl)
    #expect(imported.requiresShaderInt16)
}

#if canImport(Metal)
@Test func spirvInt16ToFloatConversionsPreserveExactBitsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(narrowIntegerFloatSPIRV(compiler)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let data = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let input: [UInt32] = [0xffff8000, 65535]
        let output = try await device.submitCommands([
            .write(buffer: data, offset: 0, words: input),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data], workgroups: .init(x: 1)),
            .copy(source: data, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        ]).wait()
        #expect(output == [input + [Float(-32768), 65535, -32768, -13, 65535, 50001].map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

private let narrowSignAndWorkgroupSource = ShaderInput.wgsl("""
    enable wgpu_int16;
    var<workgroup> signedValue: i16;
    var<workgroup> unsignedValues: array<u16, 4>;
    @group(0) @binding(0) var<storage, read_write> output: array<vec4i, 4>;
    @compute @workgroup_size(4) fn main(@builtin(local_invocation_index) lane: u32) {
        let pair = vec2<i16>(signedValue, i16(lane) - i16(2));
        output[lane] = vec4i(vec2i(sign(pair)), i32(sign(signedValue)), i32(unsignedValues[lane]));
    }
    """)

@Test func metalInt16SignUsesTypedScalarAndVectorComparisons() throws {
    let shader = try IndependentShaderCompiler().compileCompute(narrowSignAndWorkgroupSource, to: .msl)
    guard case .text(.msl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains(" > short(0)") && source.contains(" < short(0)"))
    #expect(source.contains(" > short2(0)") && source.contains(" < short2(0)"))
    #expect(!source.contains(" = sign("))
}

@Test(arguments: [ShaderOutputFormat.glsl, .hlsl])
func nativeInt16WorkgroupInitializationUsesTypedIntegerZero(_ format: ShaderOutputFormat) throws {
    let shader = try IndependentShaderCompiler().compileCompute(narrowSignAndWorkgroupSource, to: format)
    guard case .text(_, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("saga_shared_0 = int16_t(0);"))
    #expect(source.contains("saga_shared_1[saga_shared_1_init] = uint16_t(0);"))
}
