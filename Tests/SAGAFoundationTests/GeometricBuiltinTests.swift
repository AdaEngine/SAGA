import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let geometricShader = """
    const_assert dot(vec3(1, 2, 3), vec3(3, 2, 1)) == 10;
    const_assert length(vec2(3, 4)) == 5.0;
    const_assert cross(vec3f(1, 0, 0), vec3f(0, 1, 0)).z == 1.0f;
    const_assert dot(vec2i(2147483647, 1), vec2i(2, 1)) == -1i;
    const_assert dot4I8Packed(0x7f00ff80u, 0x01020304u) == -388i;
    const_assert dot4U8Packed(0x7f00ff80u, 0x01020304u) == 1404u;
    const_assert refract(vec2f(1, 0), vec2f(0, 1), 2.0f).x == 0.0f;
    @group(0) @binding(0) var<storage, read> input: array<vec4u, 4>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 7>;
    @compute @workgroup_size(1) fn run() {
        let a = bitcast<vec4f>(input[0]);
        let b = bitcast<vec4f>(input[1]);
        let axis = vec3f(0, 1, 0);
        let incidence = vec3f(0, -1, 0);
        let tangent = vec3f(1, 0, 0);
        output[0] = bitcast<vec4u>(vec4f(dot(a.xy, a.xy), length(a.xy), distance(a.xy, b.xy), distance(a.x, -2.0)));
        output[1] = bitcast<vec4u>(vec4f(cross(a.xyz, b.xyz), length(-a.y)));
        output[2] = bitcast<vec4u>(vec4f(normalize(vec3f(0, a.y, 0)), dot(a.xyz, b.xyz)));
        output[3] = bitcast<vec4u>(vec4f(faceForward(axis, incidence, axis), faceForward(axis, axis, axis).y));
        output[4] = bitcast<vec4u>(vec4f(reflect(incidence, axis), refract(incidence, axis, 0.5).y));
        output[5] = bitcast<vec4u>(vec4f(refract(tangent, axis, a.x), dot(a, b)));
        let words = input[2];
        let products = input[3];
        output[6] = vec4u(bitcast<u32>(dot(bitcast<vec4i>(words), vec4i(2, 2, 3, 4))), dot(words, vec4u(1)),
                          bitcast<u32>(dot4I8Packed(products.x, products.y)), dot4U8Packed(products.x, products.y));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsGeometricAndPackedDotBuiltins(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(geometricShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(source.contains("cross(") && source.contains("normalize(") && source.contains("refract("))
    if format == .wgsl {
        #expect(source.contains("faceForward(") && source.contains("dot4I8Packed("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    } else {
        #expect(source.contains("faceforward(") && source.contains("abs("))
    }
}

@Test(arguments: [
    "dot(1.0f, 2.0f)", "dot(vec2u(1), vec3u(2))", "dot(vec2i(1), vec2u(2))",
    "cross(vec2f(1), vec2f(2))", "normalize(vec3u(1))", "reflect(1.0f, 1.0f)",
    "refract(vec3f(1), vec3f(1), vec3f(1))", "dot4I8Packed(0i, 0u)",
])
func rejectsGeometricBuiltinShapeAndTypeMismatches(expression: String) throws {
    let compiler = IndependentShaderCompiler()
    for source in ["const result = \(expression);", "@compute @workgroup_size(1) fn run() { let result = \(expression); }"] {
        do {
            _ = try compiler.parseAndValidate(.wgsl(source))
            Issue.record("Invalid geometric builtin accepted")
        } catch ShaderFailure.diagnostics(let issues) {
            #expect(issues.first?.code == .typeMismatch)
        }
    }
}

@Test func foldsGeometricConstantsWithoutSquaringExtremeComponents() throws {
    let source = """
        const_assert length(vec2f(1e20f, 0)) == 1e20f;
        const_assert normalize(vec2f(1e-30f, 0)).x == 1.0f;
        """
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
}

private let geometricInput: [UInt32] = [
    0x40400000, 0x40800000, 0, 0, 0, 0, 0x3f800000, 0,
    0x7fffffff, 0x80000000, 0xffffffff, 2, 0x7f00ff80, 0x01020304, 0, 0,
]
private let expectedGeometry: [Float] = [
    25, 5, 5, 5, 4, -3, 0, 4, 0, 1, 0, 0,
    0, 1, 0, -1, 0, 1, 0, -1, 0, 0, 0, 0,
]
private let expectedPackedDots: [UInt32] = [3, 0, 0xfffffe7c, 1404]

#if canImport(Metal)
@Test func computesGeometricBuiltinsAndWrappingDotsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(geometricShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 112, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: geometricInput),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 112), .read(buffer: read, byteCount: 112),
        ])
        let snapshot = try #require(try await ticket.wait().first)
        #expect(snapshot.prefix(24).map { Float(bitPattern: $0) } == expectedGeometry)
        #expect(Array(snapshot.suffix(4)) == expectedPackedDots)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
