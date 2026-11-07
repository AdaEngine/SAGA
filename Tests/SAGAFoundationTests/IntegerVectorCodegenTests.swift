import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let vectorShader = """
    @group(0) @binding(1) var<storage, read> input: array<u32, 9>;
    @group(0) @binding(2) var<storage, read_write> output: array<u32, 27>;
    @compute @workgroup_size(1)
    fn run() {
        let a = vec3i(i32(input[0u]), i32(input[1u]), i32(input[2u]));
        let b = vec3i(i32(input[3u]), i32(input[4u]), i32(input[5u]));
        let count = vec3u(input[6u], input[7u], input[8u]);
        let divided = vec3u(a / b);
        let remainder = vec3u(a % b);
        let shifted = vec3u(a >> count);
        let shiftedLeft = vec3u(a << count);
        let sum = vec3u(a + 1i);
        let scale = vec3u(2i * a);
        let negated = vec3u(-a);
        let mixed = (vec3u(a) / vec3u(b)).zyx;
        let smaller = a < b;
        output[0u] = divided.x; output[1u] = divided.y; output[2u] = divided.z;
        output[3u] = remainder.x; output[4u] = remainder.y; output[5u] = remainder.z;
        output[6u] = shifted.x; output[7u] = shifted.y; output[8u] = shifted.z;
        output[9u] = shiftedLeft.x; output[10u] = shiftedLeft.y; output[11u] = shiftedLeft.z;
        output[12u] = sum.x; output[13u] = sum.y; output[14u] = sum.z;
        output[15u] = scale.x; output[16u] = scale.y; output[17u] = scale.z;
        output[18u] = negated.x; output[19u] = negated.y; output[20u] = negated.z;
        output[21u] = mixed.x; output[22u] = mixed.y; output[23u] = mixed.z;
        if smaller.x { output[24u] = 1u; }
        if smaller.y { output[25u] = 1u; }
        if smaller.z { output[26u] = 1u; }
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func lowersIntegerVectorOperationsPerLane(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(vectorShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader text"); return }
    #expect(shader.storageBindings.map(\.minimumByteSize) == [36, 108])
    #expect(source.contains(".zyx"))
    if format == .wgsl || format == .glsl {
        let input: ShaderInput = format == .wgsl ? .wgsl(source) : .glsl(source, stage: .compute, defines: [:])
        #expect(try compiler.parseAndValidate(input).module.entryPoints.first?.stage == .compute)
    }
}

@Test func rejectsInvalidIntegerVectorShapesAndStaticFaults() {
    let compiler = IndependentShaderCompiler()
    for expression in [
        "a << vec2u(0u, 1u)", "a >> vec3i(0i, 1i, 2i)",
        "a + vec2i(0i, 1i)", "a & 1i", "a == 1i",
        "a / vec3i(2i, 0i, 3i)", "a % 0i", "a << vec3u(0u, 32u, 1u)",
        "vec3i(i32(2147483648u), 0i, 2i) / vec3i(-1i, 2i, 3i)",
        "vec3u(0u, 0x80000000u, 2u) << vec3u(1u, 1u, 0u)",
    ] {
        let source = """
            @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
                let a = vec3i(i32(id.x));
                _ = \(expression);
            }
            """
        #expect(throws: ShaderFailure.self, Comment(rawValue: expression)) {
            _ = try compiler.parseAndValidate(.wgsl(source))
        }
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsVec4ArithmeticDependenciesBeforeUserHelpers(to format: ShaderOutputFormat) throws {
    let source = """
        @group(0) @binding(0) var<storage, read_write> output: u32;
        fn divide(value: vec4u, by: u32) -> vec4u { return value / by; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            let zero = 0u;
            output = divide(vec4u(id.x), zero).w;
        }
        """
    let compiler = IndependentShaderCompiler()
    guard case .text(_, let output) = try compiler.compile(.wgsl(source), to: format) else {
        Issue.record("Expected shader text"); return
    }
    let scalar = try #require(output.range(of: "saga_u32_div"))
    let vector = try #require(output.range(of: "saga_vec4_u32_divide"))
    let helper = try #require(output.range(of: "saga_fn_"))
    #expect(scalar.lowerBound < vector.lowerBound && vector.lowerBound < helper.lowerBound)
    if format == .wgsl || format == .glsl {
        let input: ShaderInput = format == .wgsl ? .wgsl(output) : .glsl(output, stage: .compute, defines: [:])
        _ = try compiler.parseAndValidate(input)
    }
}

#if canImport(Metal)
@Test func vectorIntegerEdgesExecuteThroughMetalQueue() async throws {
    let a: [Int32] = [.min, -7, .max]
    let b: [Int32] = [-1, 0, -3]
    let counts: [UInt32] = [32, 63, 33]
    let unsignedA = a.map(UInt32.init(bitPattern:))
    let unsignedB = b.map(UInt32.init(bitPattern:))
    let expected: [UInt32] =
        [0x80000000, 0xfffffff9, 0xd5555556] +
        [0, 0, 1] +
        [0x80000000, .max, 0x3fffffff] +
        [0x80000000, 0x80000000, 0xfffffffe] +
        [0x80000001, 0xfffffffa, 0x80000000] +
        [0, 0xfffffff2, 0xfffffffe] +
        [0x80000000, 7, 0x80000001] +
        [0, 0xfffffff9, 0] +
        [1, 1, 0]
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(vectorShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 36, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 108, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 108, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: unsignedA + unsignedB + counts),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 1): input, .init(binding: 2): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 108),
            .read(buffer: read, byteCount: 108),
        ])
        #expect(try await ticket.wait() == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
