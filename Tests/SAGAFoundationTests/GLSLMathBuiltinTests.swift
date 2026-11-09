import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let glslMathInput = ShaderInput.glsl("""
    #version 450
    layout(local_size_x = 1) in;
    layout(std430, set = 0, binding = 0) buffer Result { vec4 values[7]; } outputData;
    void main() {
        outputData.values[0] = round(vec4(0.5, 1.5, -1.5, 2.5));
        outputData.values[1] = clamp(vec4(-1, 0.5, 3, 1), 0.0, 1.0);
        outputData.values[2] = vec4(mod(-2.5, 2.0), mod(2.5, -2.0), ldexp(0.5, 3),
                                    smoothstep(0.0, 2.0, vec2(1.0, 2.0)).x);
        mat2 basis = mat2(4, 2, 2, 2);
        mat2 reciprocal = inverse(basis);
        outputData.values[3] = vec4(reciprocal[0], reciprocal[1]);
        outputData.values[4] = vec4(float(bitCount(0x80000001u)), float(bitfieldReverse(1u) >> 31u),
                                    determinant(basis), dot(vec2(2, 3), vec2(4, 5)));
        outputData.values[5] = vec4(unpackHalf2x16(packHalf2x16(vec2(0.5, -2.0))),
                                    float(findLSB(0u)), float(findMSB(-2)));
        outputData.values[6] = vec4(float(bitfieldExtract(-2, 0, 3)), float(bitfieldInsert(1u, 3u, 4, 2)),
                                    float(findMSB(0x80000000u)), float(findLSB(16u)));
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslMathBuiltinsReachTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(glslMathInput, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [112])
    guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

@Test func glslMathBuiltinsPreserveSPIRVAndWebGPUShape() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(glslMathInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let direct = try compiler.compileWebGPUCompute(glslMathInput)
    let imported = try compiler.compileWebGPUCompute(.spirv(words))
    for shader in [direct, imported] {
        #expect(shader.storageBindings.map(\.minimumByteSize) == [112])
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

@Test(arguments: ["inverse(mat2x3(1.0))", "clamp(vec3(0), vec2(0), 1.0)", "round(true)",
                  "ldexp(vec2(1), 2)", "bitCount(1.0)", "smoothstep(0.0, vec2(1), vec2(0))",
                  "packHalf2x16(vec3(1))", "bitfieldInsert(1u, uvec2(1u), 2, 3)", "bitfieldExtract(1u, 0u, 3)"])
func glslMathBuiltinsRejectInvalidShapesAndDomains(_ expression: String) {
    let input = ShaderInput.glsl("#version 450\nvoid main() { \(expression); }", stage: .compute, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}

#if canImport(Metal)
@Test func glslMathBuiltinsPreserveNumericResultsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(glslMathInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 112), .read(buffer: readback, byteCount: 112),
        ]).wait()
        let expected: [Float] = [0, 2, -2, 2, 0, 0.5, 1, 1, 1.5, -1.5, 4, 0.5,
                                 0.5, -0.5, -0.5, 1, 2, 1, 4, 23, 0.5, -2, -1, 0, -2, 49, 31, 4]
        #expect(snapshots == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
