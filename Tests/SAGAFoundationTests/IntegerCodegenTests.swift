import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let arithmeticShader = """
    @group(0) @binding(1) var<storage, read> input: array<u32, 18>;
    @group(0) @binding(2) var<storage, read_write> output: array<u32, 72>;
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        let source = id.x * 3u;
        let target = id.x * 12u;
        let a = input[source];
        let b = input[source + 1u];
        let count = input[source + 2u];
        var lhs = i32(a);
        let rhs = i32(b);
        output[target] = u32(lhs + rhs);
        output[target + 1u] = u32(lhs - rhs);
        output[target + 2u] = u32(lhs * rhs);
        output[target + 3u] = u32(-lhs);
        output[target + 4u] = u32(lhs / rhs);
        output[target + 5u] = u32(lhs % rhs);
        output[target + 6u] = u32(lhs << count);
        output[target + 7u] = u32(lhs >> count);
        output[target + 8u] = a / b;
        output[target + 9u] = a % b;
        output[target + 10u] = a << count;
        output[target + 11u] = a >> count;
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func lowersIntegerArithmeticWithoutNativeUndefinedBehavior(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(arithmeticShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader text"); return }
    if format == .wgsl || format == .glsl {
        let input: ShaderInput = format == .wgsl ? .wgsl(source) : .glsl(source, stage: .compute, defines: [:])
        #expect(try compiler.parseAndValidate(input).module.entryPoints.first?.stage == .compute)
    }
    if format != .wgsl {
        #expect(source.contains("& 31u"))
        #expect(source.contains("saga_i32_div"))
        #expect(source.contains("saga_i32_rem"))
        #expect(source.contains("saga_u32_div"))
        #expect(source.contains("saga_u32_rem"))
    }
}

@Test func rejectsStaticIntegerFaultsInRuntimeWGSLExpressions() {
    let compiler = IndependentShaderCompiler()
    for expression in ["id.x / (2u - 2u)", "i32(id.x) % 0i", "id.x >> 32u",
                       "i32(id.x) << 2i", "1u << 31u << 1u", "id.x + 1u << 1u",
                       "id.x & 1u << 2u", "id.x << 2u & 1u",
                       "(-2147483647i - 1i) / (-1i)", "0x80000000u << 1u"] {
        let source = "@compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) { _ = \(expression); }"
        #expect(throws: ShaderFailure.self) { _ = try compiler.parseAndValidate(.wgsl(source)) }
    }
}

@Test func parenthesizedShiftCanBeUsedInsideBitwiseExpression() throws {
    let source = """
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
            _ = id.x & (1u << 2u);
            _ = (id.x << 2u) & 1u;
        }
        """
    _ = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl)
}

@Test func regeneratedWGSLSignedLiteralsKeepTheirConcreteType() throws {
    let source = """
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(1) fn run() {
            if 2147483647i + 1i > 0i { output = 1u; }
            else { output = 0u; }
        }
        """
    let compiler = IndependentShaderCompiler()
    guard case .text(.wgsl, let output) = try compiler.compile(.wgsl(source), to: .wgsl) else {
        Issue.record("Expected WGSL"); return
    }
    #expect(output.contains("2147483647i + 1i"))
    _ = try compiler.parseAndValidate(.wgsl(output))
}

@Test func zeroInRuntimeLetDoesNotBecomeAConstDivisor() throws {
    let source = """
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
            let zero = 0u;
            _ = id.x / zero;
        }
        """
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileCompute(.wgsl(source), to: .msl)
    guard case .text(.wgsl, let emitted) = try compiler.compile(.wgsl(source), to: .wgsl) else {
        Issue.record("Expected WGSL"); return
    }
    _ = try compiler.parseAndValidate(.wgsl(emitted))
}

#if canImport(Metal)
@Test func integerEdgeCasesExecuteThroughMetalQueue() async throws {
    let rows: [(Int32, Int32, UInt32)] = [
        (.min, -1, 32), (.min, 0, 33), (.max, 1, 63), (-7, 3, 4), (7, -3, .max), (-1, -1, 0),
    ]
    var inputs: [UInt32] = [], expected: [UInt32] = []
    for (lhs, rhs, count) in rows {
        let a = UInt32(bitPattern: lhs), b = UInt32(bitPattern: rhs)
        let shift = count & 31
        let unsafeDivision = rhs == 0 || (lhs == .min && rhs == -1)
        inputs += [a, b, count]
        let signed: [Int32] = [
            lhs &+ rhs, lhs &- rhs, lhs &* rhs, 0 &- lhs,
            unsafeDivision ? lhs : lhs / rhs, unsafeDivision ? 0 : lhs % rhs,
            lhs << shift, lhs >> shift,
        ]
        expected += signed.map(UInt32.init(bitPattern:))
        expected += [b == 0 ? a : a / b, b == 0 ? 0 : a % b, a << shift, a >> shift]
    }
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(arithmeticShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 72, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 288, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 288, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: inputs),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 1): input, .init(binding: 2): output],
                             workgroups: .init(x: 6)),
            .copy(source: output, destination: read, byteCount: 288),
            .read(buffer: read, byteCount: 288),
        ])
        #expect(try await ticket.wait() == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
