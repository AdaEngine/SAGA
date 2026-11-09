import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvGeometryInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read> input: array<vec4f, 4>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 11>;

    @compute @workgroup_size(1) fn run() {
        let a = input[0];
        let b = input[1];
        let axes = input[2];
        let n = vec3f(axes.x, axes.y, axes.x);
        let incidence = vec3f(axes.x, axes.z, axes.x);
        let tangent = vec3f(axes.y, axes.x, axes.x);
        let transmitted = refract(incidence, n, axes.y * 0.5f);
        output[0] = vec4f(length(a.xy), distance(a.xy, b.xy), length(-a.y), distance(a.x, -axes.w));
        output[1] = vec4f(cross(a.xyz, b.xyz), dot(a.xy, a.xy));
        output[2] = vec4f(normalize(vec3f(axes.x, a.y, axes.x)), faceForward(n, n, n).y);
        output[3] = vec4f(faceForward(n, incidence, n), reflect(incidence, n).y);
        output[4] = vec4f(refract(tangent, n, axes.w), transmitted.y);

        let values = input[3];
        let exponentParts = frexp(values);
        let integralParts = modf(values.xy);
        let scalarExponent = frexp(values.z);
        let scalarIntegral = modf(values.x);
        output[5] = exponentParts.fract;
        output[6] = vec4f(exponentParts.exp);
        output[7] = ldexp(exponentParts.fract, exponentParts.exp);
        output[8] = vec4f(integralParts.fract, integralParts.whole);
        output[9] = vec4f(scalarExponent.fract, f32(scalarExponent.exp), scalarIntegral.fract, scalarIntegral.whole);
        output[10] = vec4f(transmitted, ldexp(scalarExponent.fract, scalarExponent.exp));
    }
    """)

private let spirvHalfNormInput = ShaderInput.wgsl("""
    enable f16;
    @group(0) @binding(0) var<storage, read> input: vec4f;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 3>;
    @compute @workgroup_size(1) fn run() {
        let large = vec2h(input.xy);
        let small = vec2h(input.zw);
        let origin = vec2h(input.yy);
        output[0] = vec4f(f32(length(large)), f32(distance(large, origin)), vec2f(normalize(large)));
        output[1] = vec4f(f32(length(small)), f32(distance(small, origin)), vec2f(normalize(small)));
        let parts = frexp(vec2h(input.xw));
        output[2] = vec4f(vec2f(parts.fract), vec2f(parts.exp));
    }
    """)

private func exportGeometry(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvGeometryInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvGeometryCodeGeneratorUsesStructureResultsForDecomposition() throws {
    let words = try exportGeometry(IndependentShaderCompiler())
    var extendedOpcodes: [UInt32: Int] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        if (words[offset] & 0xffff) == 12 && count >= 5 {
            extendedOpcodes[words[offset + 4], default: 0] += 1
        }
        offset += count
    }
    #expect(extendedOpcodes[36] == 2) // ModfStruct
    #expect(extendedOpcodes[52] == 2) // FrexpStruct
    #expect(extendedOpcodes[53] == 2) // Ldexp
    #expect(Set(extendedOpcodes.keys).isSuperset(of: 66...72))
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvGeometryCodeGeneratorRoundTripsNativeShaderFormats(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportGeometry(compiler)), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64, 176])
    guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvGeometryCodeGeneratorPreservesGeometryAndDecompositionOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shaderInput = try roundTrip ? ShaderInput.spirv(exportGeometry(compiler)) : spirvGeometryInput
    let shader = try compiler.compileCompute(shaderInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 176, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 176, usage: [.mapRead, .copyDestination]))
        let seeds: [Float] = [3, 4, 0, 0, 0, 0, 1, 0, 0, 1, -1, 2, -3.25, 2.75, -0.125, 8]
        let expected: [Float] = [
            5, 5, 4, 5, 4, -3, 0, 25, 0, 1, 0, -1, 0, 1, 0, 1, 0, 0, 0, -1,
            -0.8125, 0.6875, -0.5, 0.5, 2, 2, -2, 4, -3.25, 2.75, -0.125, 8,
            -0.25, 0.75, -3, 2, -0.5, -2, -0.25, -3, 0, -1, 0, -0.125,
        ]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: seeds.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 176), .read(buffer: readback, byteCount: 176),
        ]).wait()
        #expect(snapshots.map { $0.map(Float.init(bitPattern:)) } == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(arguments: [false, true])
func spirvGeometryCodeGeneratorPreservesHalfNormsOnMetal(_ roundTrip: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shaderInput: ShaderInput
    if roundTrip {
        guard case .spirv(let words) = try compiler.compile(spirvHalfNormInput, to: .spirv) else {
            throw ShaderFailure.incompatibleArtifact
        }
        shaderInput = .spirv(words)
    } else {
        shaderInput = spirvHalfNormInput
    }
    let shader = try compiler.compileCompute(shaderInput, to: .msl)
    #expect(shader.requiresShaderF16)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
        let small: Float = 0x1p-14
        let seeds: [Float] = [512, 0, 0, small]
        let expected: [Float] = [512, 512, 1, 0, small, small, 0, 1, 0.5, 0.5, 10, -13]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: seeds.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
        ]).wait()
        #expect(snapshots.map { $0.map(Float.init(bitPattern:)) } == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
