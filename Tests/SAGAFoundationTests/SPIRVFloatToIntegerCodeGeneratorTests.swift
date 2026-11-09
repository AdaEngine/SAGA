import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvFloatToIntegerInput = ShaderInput.wgsl("""
    enable f16;
    struct Inputs { wideUnsigned: vec4f, wideSigned: vec4f, narrow: vec4h }
    @group(0) @binding(0) var<storage, read> inputs: Inputs;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 7>;
    @compute @workgroup_size(1) fn run() {
        let a = inputs.wideUnsigned;
        let b = inputs.wideSigned;
        let h = inputs.narrow;
        output[0] = vec4u(a);
        output[1] = bitcast<vec4u>(vec4i(a));
        output[2] = bitcast<vec4u>(vec4i(b));
        output[3] = vec4u(h);
        output[4] = bitcast<vec4u>(vec4i(h));
        output[5] = vec4u(u32(a.x), bitcast<u32>(i32(b.x)), bitcast<u32>(i32(b.y)), u32(h.x));
        output[6] = vec4u(u32(a.y), bitcast<u32>(i32(a.z)), u32(h.y), bitcast<u32>(i32(h.z)));
    }
    """)

private func exportFloatToInteger(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvFloatToIntegerInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvFloatToIntegerCodeGeneratorRoundTripsScalarAndVectorCasts(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportFloatToInteger(compiler)
    try #require(words.count > 5)
    var opcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        opcodes.insert(words[offset] & 0xffff)
        offset += count
    }
    #expect(opcodes.isSuperset(of: [109, 110])) // OpConvertFToU and OpConvertFToS
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.requiresShaderF16)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [48, 112])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.shaderFloat16])
    }
}

#if canImport(Metal)
@Test func spirvFloatToIntegerCodeGeneratorSaturatesRepresentableBoundsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportFloatToInteger(compiler)), to: .msl)
    let inputWords: [UInt32] = [
        0x4f7fffff, 0x7f800000, 0xff800000, Float(-3.75).bitPattern,
        0x4effffff, 0xcf000000, Float(1.875).bitPattern, Float(-1.875).bitPattern,
        0x7c007bff, 0xc380fc00, 0xfeedface, 0xdeadbeef,
    ]
    let expected: [UInt32] = [
        4294967040, 4294967040, 0, 0,
        2147483520, 2147483520, 0x80000000, 0xfffffffd,
        2147483520, 0x80000000, 1, .max,
        65504, 65504, 0, 0,
        65504, 65504, 0xffff0020, 0xfffffffd,
        4294967040, 2147483520, 0x80000000, 65504,
        4294967040, 0x80000000, 65504, 0xffff0020,
    ]
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 48, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: inputWords),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 112), .read(buffer: readback, byteCount: 112),
        ]).wait()
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
