import SAGACompiler
import SAGAShaderIR
import Testing

#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let matrixOperatorSource = ShaderInput.glsl("""
    #version 450
    layout(local_size_x=1) in;
    layout(set=0,binding=0,std430) buffer Results { vec4 values[10]; } result;
    uint leftCalls;
    uint rightCalls;
    mat2x3 makeLeft() {
        leftCalls += 1u;
        return mat2x3(1, 2, 3, 4, 5, 6);
    }
    mat4x2 makeRight() {
        rightCalls += 1u;
        return mat4x2(7, 8, 9, 10, 11, 12, 13, 14);
    }
    void main() {
        leftCalls = 0u;
        rightCalls = 0u;
        mat2x3 a = mat2x3(1, 2, 3, 4, 5, 6);
        mat4x3 product = makeLeft() * makeRight();
        result.values[0] = vec4(product[0], product[3].z);
        result.values[1] = vec4(product[3], 1);
        result.values[2] = vec4(a * vec2(2, 3), 1);
        result.values[3] = vec4(vec3(7, 8, 9) * a, float(leftCalls), float(rightCalls));
        result.values[4] = vec4((-(2 * a))[0], 1);
        result.values[5] = vec4((a * 3u)[1], 1);
        result.values[6] = vec4((4u - a)[1], 1);
        mat2x3 component = (a + 2) / mat2x3(1, 2, 5, 2, 7, 4);
        result.values[7] = vec4(component[0], 1);
        result.values[8] = vec4((a / 2u)[1], 1);
        mat2x3 unchanged = a;
        unchanged += a;
        unchanged -= a;
        result.values[9] = vec4((24 / a)[1].xz, unchanged[0].xy);
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslMatrixOperatorsReachTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(matrixOperatorSource, entryPoint: "main", to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [160])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [
    "mat2x3 value = mat2x3(1.0) * mat2x3(2.0);",
    "vec3 value = mat2x3(1.0) * vec3(2.0);",
    "vec2 value = vec2(1.0) * mat2x3(2.0);",
    "mat2x3 value = mat2x3(1.0) + mat3x2(2.0);",
    "mat2x3 value = mat2x3(1.0) / mat3x2(2.0);",
    "mat2x3 value = mat2x3(1.0) * true;",
])
func glslMatrixOperatorsRejectIncompatibleOperands(_ body: String) {
    let source = ShaderInput.glsl("#version 450\nvoid main() { \(body) }", stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileRenderStage(source, entryPoint: "main", to: .wgsl)
    }
}

#if canImport(Metal)
@Test func glslMatrixOperatorsKeepColumnOrderAndEvaluateOperandsOnceOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(matrixOperatorSource, entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 160, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 160, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 160), .read(buffer: readback, byteCount: 160),
        ]).wait()
        let expected: [Float] = [
            39, 54, 69, 123, 69, 96, 123, 1, 14, 19, 24, 1, 50, 122, 1, 1,
            -2, -4, -6, 1, 12, 15, 18, 1, 0, -1, -2, 1, 3, 2, 1, 1,
            2, 2.5, 3, 1, 6, 4, 1, 2,
        ]
        #expect(words == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
