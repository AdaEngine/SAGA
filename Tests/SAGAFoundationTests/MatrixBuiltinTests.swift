import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let matrixBuiltinShader = """
    const FLIP = transpose(mat2x3(1, 2, 3, 4, 5, 6));
    const_assert FLIP[2][1] == 6.0;
    const_assert transpose(FLIP)[1][0] == 4.0;
    const_assert determinant(mat3x3(1, 0, 5, 2, 1, 6, 3, 4, 0)) == 1.0;
    const_assert determinant(mat3x3f(1, 2, 3, 1, 2, 3, 0, 0, 1)) == 0.0f;
    const_assert determinant(mat4x4f(0, 2, 0, 0, 3, 0, 0, 0, 1, 2, 4, 0, 5, 6, 7, -1)) == 24.0f;
    @group(0) @binding(0) var<storage, read> input: array<vec4f, 9>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 3>;
    @group(0) @binding(2) var<storage, read_write> columns: array<mat3x2f, 1>;
    fn flip(m: mat2x3f) -> mat3x2f { return transpose(m); }
    @compute @workgroup_size(1) fn run() {
        let rect = mat2x3f(input[0].xyz, input[1].xyz);
        let square = mat2x2f(input[0].xy, input[1].xy);
        let m3 = mat3x3f(input[2].xyz, input[3].xyz, input[4].xyz);
        let m4 = mat4x4f(input[5], input[6], input[7], input[8]);
        let turned = flip(rect);
        let restored = transpose(turned);
        const local = determinant(mat2x2f(1, 2, 3, 4));
        columns[0] = turned;
        output[0] = vec4f(determinant(square), determinant(m3), determinant(m4), local);
        output[1] = vec4f(restored[0], turned[2].y);
        output[2] = vec4f(turned * vec3f(1, 2, 3), restored[1].xy);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsTransposeAndDeterminantForMatrixLayouts(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(matrixBuiltinShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(source.contains("transpose(") && source.contains("determinant("))
    switch format {
    case .wgsl:
        #expect(source.contains("mat3x2<f32>"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl: #expect(source.contains("float3x2"))
    case .hlsl: #expect(source.contains("float3x2") && source.contains("mul("))
    case .glsl: #expect(source.contains("mat3x2"))
    default: Issue.record("Unexpected shader target")
    }
}

@Test(arguments: [
    "determinant(mat2x3f())", "transpose(vec2f(1))", "transpose(mat2x2f(), mat2x2f())",
])
func rejectsInvalidMatrixBuiltins(expression: String) throws {
    let compiler = IndependentShaderCompiler()
    for source in ["const value = \(expression);", "@compute @workgroup_size(1) fn run() { let value = \(expression); }"] {
        do {
            _ = try compiler.parseAndValidate(.wgsl(source))
            Issue.record("Invalid matrix builtin accepted")
        } catch ShaderFailure.diagnostics(let issues) {
            #expect(issues.first?.code == .typeMismatch)
        }
    }
}

@Test func determinantConstRejectsNonFiniteF32Result() throws {
    let source = "const value = determinant(mat2x2f(1e30f, 0, 0, 1e30f));"
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Non-finite determinant accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .invalidType)
    }
}

private let matrixBuiltinInput: [Float] = [
    1, 2, 3, 0, 4, 5, 6, 0,
    1, 0, 5, 0, 2, 1, 6, 0, 3, 4, 0, 0,
    0, 2, 0, 0, 3, 0, 0, 0, 1, 2, 4, 0, 5, 6, 7, -1,
]

#if canImport(Metal)
@Test func evaluatesMatrixBuiltinsAndStoresTransposedColumnsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(matrixBuiltinShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 144, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let columns = try await device.createBuffer(.init(size: 24, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: matrixBuiltinInput.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output,
                             .init(binding: 2): columns], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
            .copy(source: columns, destination: read, byteCount: 24), .read(buffer: read, byteCount: 24),
        ])
        let expected: [[Float]] = [[-3, 1, 24, -2, 1, 2, 3, 6, 14, 32, 4, 5], [1, 4, 2, 5, 3, 6]]
        #expect(try await ticket.wait() == expected.map { $0.map(\.bitPattern) })
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
