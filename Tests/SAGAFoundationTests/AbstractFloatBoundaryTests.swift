import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let boundaryShader = """
    enable f16;
    const positive: f32 = 3.40282347e38;
    const halfMaximum: f16 = 65519.0;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;

    @compute @workgroup_size(1)
    fn run() {
        let negative: f32 = -3.40282347e38;
        let halfMinimum: f16 = -65519;
        output[0] = bitcast<u32>(positive);
        output[1] = bitcast<u32>(negative);
        output[2] = bitcast<u32>(vec2h(halfMaximum, halfMinimum));
        let consistent = (1.5 + 2) == (2 + 1.5);
        output[3] = select(0u, bitcast<u32>(f32(3.40282347e38)), consistent);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func nagaAbstractFloatBoundariesRoundBeforeCheckingOverflow(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler(nagaFloatRounding: true)
    let shader = try compiler.compileCompute(.wgsl(boundaryShader), to: format)
    #expect(shader.requiresShaderF16)
    guard case .text(let actual, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl {
        #expect(code.contains("-0x1.fffffep+127f"))
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(code), capabilities: [.shaderFloat16])
    }
}

@Test(arguments: ["const v: f32 = 3.40282347e38;", "const v: f16 = 65519.0;"])
func strictAbstractFloatBoundariesRejectOutsideFiniteRange(_ declaration: String) {
    let input = ShaderInput.wgsl("enable f16; " + declaration + " @compute @workgroup_size(1) fn run() {}")
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compileCompute(input, to: .wgsl) }
}

@Test(arguments: [
    "const v: f32 = 3.40282357e38;",
    "const v: f16 = 65520.0;",
    "const v = quantizeToF16(65505.0f);",
    "const v = pack2x16float(vec2f(0.0f, 65505.0f));",
])
func nagaAbstractFloatBoundariesRejectOverflowAndKeepPackingChecks(_ declaration: String) {
    let input = ShaderInput.wgsl("enable f16; " + declaration + " @compute @workgroup_size(1) fn run() {}")
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler(nagaFloatRounding: true).compileCompute(input, to: .msl)
    }
}

#if canImport(Metal)
@Test func nagaAbstractFloatBoundariesPreserveFiniteBitsOnMetal() async throws {
    let shader = try IndependentShaderCompiler(wgslSourceProfile: .nagaNative)
        .compileCompute(.wgsl(boundaryShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [[0x7f7fffff, 0xff7fffff, 0xfbff7bff, 0x7f7fffff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
