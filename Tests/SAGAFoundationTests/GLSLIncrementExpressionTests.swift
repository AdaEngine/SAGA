import SAGACompiler
import SAGAShaderIR
import Testing

#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let incrementExpressionSource = ShaderInput.glsl("""
    #version 450
    layout(local_size_x=1) in;
    layout(set=0,binding=0,std430) buffer Results { int cells[3]; vec4 values[9]; } result;
    uint indexCalls;
    int nextIndex() {
        indexCalls += 1u;
        return 1;
    }
    void main() {
        indexCalls = 0u;
        result.cells[0] = 10;
        result.cells[1] = 20;
        result.cells[2] = 30;
        int indexedOld = result.cells[nextIndex()]++;
        int cursor = 0;
        int indexedNew = --result.cells[cursor++];

        int scalar = 5;
        int oldScalar = scalar++ + 0;
        int newScalar = --scalar + 0;
        float fractional = 1.25;
        float oldFractional = fractional++;
        float newFractional = --fractional;
        result.values[0] = vec4(oldScalar, newScalar, oldFractional, newFractional);

        uvec2 pair = uvec2(1u, 3u);
        uvec2 oldPair = pair--;
        uvec2 newPair = ++pair;
        result.values[1] = vec4(oldPair, newPair);

        vec3 components = vec3(3, 5, 7);
        vec2 oldSwizzle = components.yx++;
        vec2 newSwizzle = --components.xy;
        result.values[2] = vec4(oldSwizzle, newSwizzle);
        result.values[3] = vec4(components, float(indexCalls));

        mat4x3 matrix = mat4x3(1.0);
        mat4x3 oldMatrix = matrix++;
        mat4x3 incremented = matrix;
        mat4x3 restored = --matrix;
        result.values[4] = vec4(oldMatrix[3], incremented[3].x);
        result.values[5] = vec4(incremented[0], restored[0].x);
        result.values[6] = vec4(restored[3], restored[1].y);
        result.values[7] = vec4(indexedOld, indexedNew, cursor, float(indexCalls));

        uint wrapped = 0u;
        uint beforeWrap = wrapped--;
        bool skipped = false && (++scalar == 6);
        result.values[8] = vec4(float(beforeWrap == 0u), float(wrapped == 0xffffffffu), float(skipped), float(scalar));
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslIncrementExpressionsReachTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(incrementExpressionSource, entryPoint: "main", to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [160])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [
    "bool value = true; value++;",
    "float values[2]; --values;",
    "const int value = 1; value++;",
    "vec3 value = vec3(1); value.xx++;",
    "int value = 1; (value + 1)++;",
    "int value = 1; ++(value++);",
    "int value = 1; (++value) = 3;",
])
func glslIncrementExpressionsRequireWritableNumericLvalues(_ body: String) {
    let source = ShaderInput.glsl("#version 450\nvoid main() { \(body) }", stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileRenderStage(source, entryPoint: "main", to: .wgsl)
    }
}

#if canImport(Metal)
@Test func glslIncrementExpressionsPreserveOldNewValuesAndAddressesOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(incrementExpressionSource, entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 160, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 160, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 160), .read(buffer: readback, byteCount: 160),
        ]).wait()
        let words = try #require(snapshots.first)
        try #require(words.count == 40)
        #expect(Array(words.prefix(3)) == [9, 21, 30])
        let expected: [Float] = [
            5, 5, 1.25, 1.25, 1, 3, 1, 3, 5, 3, 3, 5, 3, 5, 7, 1,
            0, 0, 0, 1, 2, 1, 1, 1, 0, 0, 0, 1, 20, 9, 1, 1, 1, 1, 0, 5,
        ]
        #expect(Array(words.dropFirst(4)) == expected.map(\.bitPattern))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
