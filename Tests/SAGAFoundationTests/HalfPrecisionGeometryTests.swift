import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let halfGeometryShader = """
    enable f16;
    const_assert dot(vec2h(1, 0x1p-11h), vec2(1.0, 1.0)) == 1h;
    const_assert length(vec2h(60000, 0)) == 60000h;
    const_assert normalize(vec2h(0x1p-14h, 0)).x == 1h;
    const_assert distance(3h, -2) == 5h;
    const_assert cross(vec3h(1, 0, 0), vec3(0.0, 1.0, 0.0)).z == 1h;
    const_assert reflect(vec2h(0, -1), vec2h(0, 1)).y == 1h;
    const_assert refract(vec2h(1, 0), vec2h(0, 1), 2.0).x == 0h;
    const_assert faceForward(vec2h(0, 1), vec2h(0, -1), vec2h(0, 1)).y == 1h;
    @group(0) @binding(0) var<storage, read> input: array<vec4h, 4>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 7>;
    @compute @workgroup_size(1) fn run() {
        let a = input[0];
        let b = input[1];
        let axis = input[2].xyz;
        let settings = input[3];
        let incidence = vec3h(0, -axis.y, 0);
        let tangent = vec3h(1, 0, 0);
        output[0] = vec4f(f32(dot(a.xy, a.xy)), f32(length(a.xy)), f32(distance(a.xy, b.xy)), f32(distance(a.x, -2.0)));
        output[1] = vec4f(vec3f(cross(a.xyz, b.xyz)), f32(length(-a.y)));
        output[2] = vec4f(vec3f(normalize(vec3h(0, a.y, 0))), f32(dot(a.xyz, b.xyz)));
        output[3] = vec4f(vec3f(faceForward(axis, incidence, axis)), f32(faceForward(axis, axis, axis).y));
        output[4] = vec4f(vec3f(reflect(incidence, axis)), f32(refract(incidence, axis, settings.z).y));
        output[5] = vec4f(vec3f(refract(tangent, axis, settings.y)), f32(dot(a, b)));
        let wide = vec3h(settings.x, 0, 0);
        let tiny = vec3h(0, settings.w, 0);
        output[6] = vec4f(f32(length(wide)), f32(normalize(wide).x),
                          f32(normalize(tiny).y), f32(distance(wide, vec3h(0))));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsHalfPrecisionGeometry(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(halfGeometryShader), to: format)
    #expect(shader.requiresShaderF16)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [32, 112])
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(body.contains("normalize(") && body.contains("cross(") && body.contains("refract("))
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(body), capabilities: [.shaderFloat16])
    }
}

@Test(arguments: [
    ("dot(vec2h(1), vec2f(1))", ShaderDiagnosticCode.typeMismatch),
    ("cross(vec2h(1), vec2h(1))", .typeMismatch),
    ("refract(vec3h(1), vec3h(1), 1.0f)", .typeMismatch),
    ("faceForward(vec2h(1), vec3h(1), vec2h(1))", .typeMismatch),
    ("normalize(vec3h(0))", .invalidType),
    ("dot(vec2h(300), vec2h(300))", .invalidType),
])
func rejectsInvalidHalfPrecisionGeometry(expression: String, code: ShaderDiagnosticCode) throws {
    for source in [
        "enable f16; const value = \(expression);",
        "enable f16; @compute @workgroup_size(1) fn run() { let value = \(expression); }",
    ] {
        do {
            _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
            Issue.record("Invalid half-precision geometry accepted")
        } catch ShaderFailure.diagnostics(let issues) {
            #expect(issues.first?.code == code)
        }
    }
}

#if canImport(Metal)
@Test func evaluatesHalfPrecisionGeometryOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfGeometryShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 112, usage: [.mapRead, .copyDestination]))
        let data: [Double] = [3, 4, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 60000, 2, 0.5, 0x1p-14]
        let words = stride(from: 0, to: data.count, by: 2).map { i in
            UInt32(Binary16.encode(data[i])) | (UInt32(Binary16.encode(data[i + 1])) << 16)
        }
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: words),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 112), .read(buffer: read, byteCount: 112),
        ])
        let snapshot = try #require(try await ticket.wait().first)
        let expected: [Float] = [
            25, 5, 5, 5, 4, -3, 0, 4, 0, 1, 0, 0,
            0, 1, 0, -1, 0, 1, 0, -1, 0, 0, 0, 0, 60000, 1, 1, 60000,
        ]
        #expect(snapshot.map { Float(bitPattern: $0) } == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
