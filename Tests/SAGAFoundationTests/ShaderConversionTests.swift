import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let conversionShader = """
    override gain = 0.0f;
    override width = 1u + u32(bool(gain));
    @group(0) @binding(0) var<storage, read> input: array<vec4f, 1>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;
    @compute @workgroup_size(width)
    fn run(@builtin(local_invocation_index) id: u32) {
        if id != 0u { return; }
        let samples = input[0];
        let mask = vec4<bool>(samples);
        output[0] = bitcast<vec4u>(vec4i(samples));
        output[1] = vec4u(samples);
        output[2] = vec4u(mask);
        output[3] = vec4u(u32(bool(samples.w)), u32(bool(samples.z)),
                          bitcast<u32>(f32(bool(samples.y))), bitcast<u32>(i32(bool(samples.x))));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesBooleanAndSaturatingConversions(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(conversionShader)
    let defaults = try compiler.compileCompute(source, to: format)
    let shader = try compiler.compileCompute(source, to: format, constants: ["gain": -3])
    #expect(defaults.workgroupSize == .init(x: 1))
    #expect(shader.workgroupSize == .init(x: 2))
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
}

@Test func explicitlyConvertsAbstractScalarsAndVectors() throws {
    let source = ShaderInput.wgsl("""
        const_assert bool(-2) && !bool(-0.0);
        @group(0) @binding(0) var<storage, read_write> output: array<vec2i, 2>;
        @compute @workgroup_size(1) fn run() {
            output[0] = vec2i(vec2(3.75, -3.75));
            output[1] = vec2i(i32(-2.75), i32(bool(1)));
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    #expect(generated.contains("3i") && generated.contains("-3i") && generated.contains("-2i"))
}

@Test func booleanAndNumericConversionRemainsExplicit() {
    let input = ShaderInput.wgsl("""
        @compute @workgroup_size(1) fn run() { _ = vec3u(vec2<bool>(true), 1u); }
        """)
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}

private let halfConversionShader = """
    enable f16;
    override value = 4.5h;
    override width = u32(vec2i(vec2h(value)).x) + u32(vec2<bool>(vec2h(value)).y);
    @group(0) @binding(0) var<storage, read_write> output: array<vec4i, 1>;
    @compute @workgroup_size(width)
    fn run(@builtin(local_invocation_index) id: u32) {
        if id == 0u {
            let samples = vec2h(f16(id), -2.75h);
            let mask = vec2<bool>(id == 0u, id != 0u);
            output[0] = vec4i(vec2i(samples), vec2i(vec2h(mask)));
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesHalfPrecisionIntegerAndBooleanConversions(format: ShaderOutputFormat) throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfConversionShader), to: format,
                                                                 constants: ["value": 6.75])
    #expect(shader.workgroupSize == .init(x: 7))
    #expect(shader.requiresShaderF16)
}

@Test func rejectsPipelineFloatConversionOutsideHalfFiniteRange() throws {
    let source = ShaderInput.wgsl("""
        enable f16;
        override value = 1.0f;
        @compute @workgroup_size(1) fn run() { _ = vec2h(vec2f(value, value + 1.0f)); }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileCompute(source, to: .wgsl)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, to: .wgsl, constants: ["value": 65504])
    }
}

#if canImport(Metal)
@Test func runsBooleanAndSaturatingConversionsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(conversionShader), to: .msl,
                                                                 constants: ["gain": -3])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let samples: [Float] = [-1e20, 1e20, -3.75, -0.0]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: samples.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        ])
        #expect(try await ticket.wait() == [[
            0x80000000, 0x7fffff80, 0xfffffffd, 0,
            0, 0xffffff00, 0, 0,
            1, 1, 1, 0,
            0, 1, 0x3f800000, 1,
        ]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
@Test func runsHalfPrecisionIntegerAndBooleanConversionsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfConversionShader), to: .msl,
                                                                 constants: ["value": 6.75])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[0, 0xfffffffe, 1, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
