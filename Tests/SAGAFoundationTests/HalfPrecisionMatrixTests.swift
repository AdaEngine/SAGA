import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let halfMatrixShader = """
    enable f16;
    const rounded = mat2x2h(mat2x2f(1.00048828125, 2, 3, 4));
    const_assert rounded[0][0] == 1h;
    const_assert determinant(mat2x2<f16>(1, 2, 3, 4)) == -2h;
    const_assert transpose(mat2x3(1h, 2, 3, 4, 5, 6))[2].y == 6h;
    const_assert (mat2x2h(1, 2, 3, 4) * vec2(2, 3)).y == 16h;
    struct Data { basis: mat2x3h, transforms: array<mat2x3h> }
    @group(0) @binding(0) var<storage, read_write> data: Data;
    @group(0) @binding(1) var<uniform> scale: mat2x2<f16>;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4f, 4>;
    fn flip(value: mat2x3h) -> mat3x2h { return transpose(value); }
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        let saved = data.basis;
        let turned = flip(saved);
        let gram = turned * saved;
        let widened = mat2x3f(saved);
        let adjusted = 2 * (mat2x3h(widened) + saved - saved);
        data.basis = saved * scale;
        let i = id.x + 1u;
        data.transforms[i] = data.transforms[0u] * scale;
        data.transforms[i][id.x] = vec3h(21, 22, 23);
        data.transforms[i][1u][1u] = 55h;
        data.transforms[id.x + 2u] = saved;
        let missing = data.transforms[id.x + 2u];
        output[0] = vec4f(vec3f(saved * vec2h(2, 3)), f32(determinant(scale)));
        output[1] = vec4f(vec2f(vec3h(1, 2, 3) * saved), vec2f(turned[2]));
        output[2] = vec4f(f32(missing[0].x), f32(arrayLength(&data.transforms)), f32(data.transforms[i][1].y), f32(adjusted[1].z));
        output[3] = vec4f(vec2f(gram[0]), vec2f(gram[1]));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsHalfPrecisionMatrixArithmeticAndBufferLayouts(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(halfMatrixShader), to: format)
    #expect(shader.requiresShaderF16)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [32, 64])
    #expect(shader.storageBindings.first?.runtimeArrayOffset == 16)
    #expect(shader.storageBindings.first?.runtimeArrayStride == 16)
    #expect(shader.uniformBindings.map(\.minimumByteSize) == [8])
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(body.contains("transpose(") && body.contains("determinant("))
    switch format {
    case .wgsl:
        #expect(body.contains("mat2x3<f16>"))
        _ = try compiler.parseAndValidate(.wgsl(body), capabilities: [.shaderFloat16])
    case .msl: #expect(body.contains("half2x3"))
    case .hlsl: #expect(body.contains("half2x3") && body.contains("mul(") && body.contains("Store<half>"))
    case .glsl: #expect(body.contains("f16mat2x3") && body.contains("unpackFloat2x16"))
    default: Issue.record("Unexpected shader target")
    }
}

@Test(arguments: [
    ("const m = mat2x2h();", ShaderDiagnosticCode.unsupportedCapability),
    ("alias M = mat2x3<f16>;", .unsupportedCapability),
    ("enable f16; const m = mat2x2h() * mat2x2f();", .typeMismatch),
    ("enable f16; const m = mat2x3h(mat3x2f());", .typeMismatch),
    ("enable f16; const d = determinant(mat2x3h());", .typeMismatch),
    ("enable f16; const d = determinant(mat2x2h(60000, 0, 0, 60000));", .invalidType),
    ("enable f16; @compute @workgroup_size(1) fn run() { var m = mat2x2h(); let n = m * mat2x2f(); }", .typeMismatch),
])
func rejectsInvalidHalfPrecisionMatrixExpressions(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
        Issue.record("Invalid f16 matrix accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

#if canImport(Metal)
@Test func evaluatesHalfPrecisionMatrixProductsAndPreservesPaddingOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfMatrixShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let data = try await device.createBuffer(.init(size: 52, usage: [.storage, .copyDestination, .copySource]))
        let scale = try await device.createBuffer(.init(size: 8, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        func pair(_ x: Double, _ y: Double) -> UInt32 {
            UInt32(Binary16.encode(x)) | (UInt32(Binary16.encode(y)) << 16)
        }
        let initial: [UInt32] = [
            pair(1, 2), pair(3, -1), pair(4, 5), pair(6, -2),
            pair(7, 8), pair(9, -3), pair(10, 11), pair(12, -4),
            pair(13, 14), pair(15, -5), pair(16, 17), pair(18, -6), 999,
        ]
        let expectedData: [UInt32] = [
            pair(2, 4), pair(6, -1), pair(12, 15), pair(18, -2),
            pair(7, 8), pair(9, -3), pair(10, 11), pair(12, -4),
            pair(21, 22), pair(23, -5), pair(30, 55), pair(36, -6), 999,
        ]
        let ticket = try await device.submitCommands([
            .write(buffer: data, offset: 0, words: initial),
            .write(buffer: scale, offset: 0, words: [pair(2, 0), pair(0, 3)]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data, .init(binding: 1): scale,
                             .init(binding: 2): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
            .copy(source: data, destination: read, byteCount: 52), .read(buffer: read, byteCount: 52),
        ])
        let expected: [Float] = [14, 19, 24, 6, 14, 32, 3, 6, 0, 2, 55, 12, 14, 32, 32, 77]
        #expect(try await ticket.wait() == [expected.map(\.bitPattern), expectedData])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
