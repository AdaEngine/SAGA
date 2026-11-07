import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let matrixShader = """
    @group(0) @binding(0) var<storage, read_write> output: array<f32, 18>;
    fn product(a: mat2x3f, b: mat4x2f) -> mat4x3f { return a * b; }
    fn column(a: mat2x3f, v: vec2f) -> vec3f { return a * v; }
    fn row(v: vec3f, a: mat2x3f) -> vec2f { return v * a; }
    fn seed(i: u32) -> f32 { if i == 0u { return 1.0f; } return 2.0f; }
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        let a = mat2x3f(vec3f(seed(id.x), 2.0f, 3.0f), vec3f(4.0f, 5.0f, 6.0f));
        let b = mat4x2f(vec2f(7.0f, 8.0f), vec2f(9.0f, 10.0f), vec2f(11.0f, 12.0f), vec2f(13.0f, 14.0f));
        let p = product(a, b);
        let first = p[0u];
        let last = p[id.x + 3u];
        let cv = column(a, vec2f(2.0f, 3.0f));
        let rv = row(vec3f(2.0f, 3.0f, 4.0f), a);
        let scaled = (a * 2.0f) + (3.0f * a) - a;
        var scratch: mat2x3f;
        scratch[id.x + 1u] = vec3f(7.0f, 8.0f, 9.0f);
        scratch[i32(id.x) - 1i] = vec3f(999.0f);
        let selected = scratch[id.x + 1u];
        let outside = scratch[id.x + 2u];
        output[0] = first.x; output[1] = first.y; output[2] = first.z;
        output[3] = last.x; output[4] = last.y; output[5] = last.z;
        output[6] = cv.x; output[7] = cv.y; output[8] = cv.z;
        output[9] = rv.x; output[10] = rv.y; output[11] = scaled[1u].z;
        output[12] = selected.x; output[13] = selected.y; output[14] = selected.z;
        output[15] = outside.x; output[16] = outside.y; output[17] = outside.z;
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsRectangularMatrixArithmetic(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(matrixShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .msl: #expect(source.contains("float2x3") && source.contains("float4x3"))
    case .hlsl:
        #expect(source.contains("float2x3") && source.contains("float4x3"))
        #expect(source.contains("mul(saga_param_1, saga_param_0)"))
    case .glsl: #expect(source.contains("mat2x3") && source.contains("mat4x3"))
    case .wgsl:
        #expect(source.contains("mat2x3<f32>") && source.contains("mat4x3<f32>"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target")
    }
}

#if canImport(Metal)
@Test func matrixArithmeticAndColumnBoundsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(matrixShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 72, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 72, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 72),
            .read(buffer: read, byteCount: 72),
        ])
        let expected: [Float] = [39, 54, 69, 69, 96, 123, 14, 19, 24, 20, 47, 24, 7, 8, 9, 0, 0, 0]
        #expect(try await ticket.wait() == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
