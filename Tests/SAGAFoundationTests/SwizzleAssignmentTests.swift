import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let swizzleAssignmentShader = """
    requires swizzle_assignment, pointer_composite_access;

    var<private> values: array<vec4u, 1>;
    var<private> events: u32;
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 6>;

    fn updateIndex() -> u32 {
        events = events * 10u + 1u;
        values[0] = values[0] + vec4u(2u, 4u, 8u, 10u);
        return 0u;
    }

    fn updateRHS() -> vec2u {
        events = events * 10u + 2u;
        values[0].x = 900u;
        values[0].z = 100u;
        return vec2u(55u, 66u);
    }

    @compute @workgroup_size(1) fn run() {
        values[updateIndex()].yx = updateRHS();
        output[0] = values[0];
        let simpleEvents = events;
        values[0] = vec4u();
        events = 0u;
        values[updateIndex()].yx += updateRHS();
        output[1] = values[0];
        output[2] = vec4u(simpleEvents, events, 0u, 0u);

        var local = vec4u(1u, 2u, 3u, 4u);
        let p = &local;
        p.wzyx.xy = p.xy;
        p.yxz[1u] = 9u;
        p.yxz.y += 2u;
        var invalid = -1i;
        p.yxz[invalid] = 999u;
        p.zy <<= vec2u(1u, 2u);
        p.yx = p.xy;
        output[3] = local;
        let destination = &output;
        destination[3u].yx = destination[3u].xy;

        var f = vec4f(1.0f, 2.0f, 3.0f, 4.0f);
        f.wy *= vec2f(2.0f, 3.0f);
        output[4] = bitcast<vec4u>(f);
        var b = vec4<bool>(true, true, false, true);
        b.yx &= vec2<bool>(true, false);
        b.zx |= vec2<bool>(true, false);
        b.wy.y &= false;
        output[5] = vec4u(b);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesSwizzleAssignments(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(swizzleAssignmentShader), to: format)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl {
        #expect(!generated.contains("requires swizzle_assignment"))
        _ = try compiler.parseAndValidate(.wgsl(generated))
    }
}

@Test(arguments: [
    "var v = vec4u(); v.xx = vec2u(1u);",
    "var v = vec4u(); v.xxy.xy = vec2u(1u);",
    "let v = vec4u(); v.xy = vec2u(1u);",
    "var v = vec4u(); let p = &v.xy; _ = p;",
    "var v = vec4u(); v.xy[2u] = 1u;",
    "var v = vec4u(); v.xy[i32(2.0f)] = 1u;",
    "var v = vec4u(); v.xy /= vec2u(1u, 0u);",
    "var v = vec4u(); v.xy <<= vec2u(0u, 32u);",
])
func rejectsInvalidSwizzleAssignments(statement: String) {
    let source = ShaderInput.wgsl("""
        requires swizzle_assignment;
        @compute @workgroup_size(1) fn run() { \(statement) }
        """)
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(source) }
}

@Test func rejectsOutOfRangeSwizzleOverrideIndex() throws {
    let source = ShaderInput.wgsl("""
        requires swizzle_assignment;
        override lane = 0i;
        @compute @workgroup_size(1) fn run() {
            var v = vec4u();
            v.yx[lane] = 1u;
        }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileCompute(source, to: .wgsl)
    for index in [-1.0, 2.0] {
        #expect(throws: ShaderFailure.self) {
            try compiler.compileCompute(source, to: .wgsl, constants: ["lane": index])
        }
    }
}

#if canImport(Metal)
@Test func swizzleAssignmentsPreserveEvaluationOrderOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(swizzleAssignmentShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 96, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
        ])
        #expect(try await ticket.wait() == [[
            66, 55, 100, 10, 68, 59, 100, 10, 12, 12, 0, 0,
            11, 8, 4, 1, 0x3f800000, 0x40c00000, 0x40400000, 0x41000000, 0, 0, 1, 1,
        ]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
