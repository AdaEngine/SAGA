import SAGACompiler
import SAGAShaderIR
import Testing

#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let matrixComparisonSource = ShaderInput.glsl("""
    #version 450
    layout(local_size_x=1) in;
    layout(set=0,binding=0,std430) buffer Data { vec4 values; uvec4 flags[2]; } data;
    uint leftCalls;
    uint rightCalls;
    mat2x3 leftOperand() {
        leftCalls += 1u;
        return mat2x3(vec3(data.values.x, 2, 3), vec3(4, 5, 6));
    }
    mat2x3 rightOperand() {
        rightCalls += 1u;
        return mat2x3(vec3(data.values.y, 2, 3), vec3(4, 5, 6));
    }
    void main() {
        leftCalls = 0u;
        rightCalls = 0u;
        bool same = leftOperand() == rightOperand();
        mat2x3 a = mat2x3(0, 2, 3, 4, 5, 6);
        mat2x3 b = mat2x3(0, 2, 3, 4, 5, 7);
        mat2x3 n = mat2x3(vec3(0, 2, 3), vec3(data.values.z, 5, 6));
        data.flags[0] = uvec4(uint(same), uint(a == b), uint(a != b), uint(n == n));
        data.flags[1] = uvec4(uint(n != n), leftCalls, rightCalls, uint(a != a));
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslMatrixComparisonsReachTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(matrixComparisonSource, entryPoint: "main", to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [48])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [
    "mat2x3(1.0) == mat3x2(1.0)", "mat2x3(1.0) != vec3(1.0)", "mat2x3(1.0) == 1.0",
])
func glslMatrixComparisonsRejectDifferentShapes(_ expression: String) {
    let source = ShaderInput.glsl("#version 450\nvoid main() { bool value = \(expression); }", stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileRenderStage(source, entryPoint: "main", to: .wgsl)
    }
}

#if canImport(Metal)
@Test func glslMatrixComparisonsReduceEveryComponentAndEvaluateOperandsOnceOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(matrixComparisonSource, entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let data = try await device.createBuffer(.init(size: 48, usage: [.storage, .copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
        let input: [UInt32] = [0x80000000, 0, 0x7fc00000, 0]
        let snapshots = try await device.submitCommands([
            .write(buffer: data, offset: 0, words: input),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data], workgroups: .init(x: 1)),
            .copy(source: data, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
        ]).wait()
        #expect(snapshots == [input + [1, 0, 1, 0, 1, 1, 1, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
