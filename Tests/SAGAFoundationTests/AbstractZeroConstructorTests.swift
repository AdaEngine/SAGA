import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let abstractZeroShader = """
    var<private> globalZero: vec2u = vec2();
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;

    fn sum(value: vec2u) -> u32 { return value.x + value.y + 3u; }
    fn zeroMatrix() -> mat2x2f { return mat2x2(vec2(), vec2()); }

    @compute @workgroup_size(1) fn run() {
        let inferred = vec2();
        let signed: vec2i = vec2();
        let unsigned: vec2u = vec2();
        let floating: vec2f = vec2();
        let nested: array<vec2f, 2> = array(vec2(), vec2(0, 3));
        var replaced = vec2u(17u, 19u);
        replaced = vec2();
        let matrix: mat2x2f = mat2x2();
        let returned = zeroMatrix();
        let spread: vec3u = vec3(vec2(), 7);

        output[0] = sum(vec2()) + globalZero.x + unsigned.x + replaced.x;
        output[1] = u32(nested[0].x + nested[1].y + floating.y);
        output[2] = u32(matrix[1].y + returned[0].x) + u32(inferred.x + signed.y);
        output[3] = spread.x + spread.y + spread.z;
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func abstractZeroConstructorsMaterializeInContext(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(abstractZeroShader), to: format)
    #expect(shader.storageBindings.first?.minimumByteSize == 16)
    guard case .text(let actual, let text) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

@Test(arguments: [
    "let value: vec2<bool> = vec2();",
    "let value: vec3u = vec2();",
    "let value: vec2u = vec2i();",
    "let value: mat2x3f = mat2x2();",
])
func abstractZeroConstructorsRejectIncompatibleContexts(_ declaration: String) throws {
    let shader = ShaderInput.wgsl("@compute @workgroup_size(1) fn run() { \(declaration) }")
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(shader)
        Issue.record("Accepted a zero constructor in an incompatible context")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func abstractZeroConstructorsProduceContextualValuesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(abstractZeroShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [[3, 3, 0, 7]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
