import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let pointerShader = """
    struct Packet { values: array<u32, 2>, stamp: u32, }
    var<private> seed = 4u;
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u>;

    fn mutate(packet: ptr<function, Packet>, increment: ptr<private, u32>, index: i32) -> u32 {
        let target = &(*packet).values[index];
        let old = *target;
        *target = old + *increment;
        *increment += 1u;
        (*packet).stamp += *increment;
        return old;
    }

    fn forward(packet: ptr<function, Packet>, increment: ptr<private, u32>, index: i32) -> u32 {
        let again: ptr<function, Packet> = &*packet;
        return mutate(again, increment, index);
    }

    fn readBoth(a: ptr<function, u32>, b: ptr<function, u32>) -> u32 { return *a + *b; }

    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32) {
        var packet = Packet(array(10u, 20u), 100u);
        var index = i32(lane) - 1i;
        let chosen = &packet.values[index];
        index = 0i;
        let before = *chosen;
        *chosen = before + 1u;
        let previous = forward(&packet, &seed, i32(lane) - 1i);
        var tag = packet.stamp;
        let outputAddress = &output;
        output[lane] = vec4u(before, previous, packet.values[0] + packet.values[1],
                             readBoth(&tag, &tag) + seed + arrayLength(outputAddress));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesShaderPointerParameters(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(pointerShader), to: format)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
}

@Test(arguments: [
    """
    fn update(a: ptr<function, u32>, b: ptr<function, u32>) { *a = *b; }
    fn forward(a: ptr<function, u32>, b: ptr<function, u32>) { update(a, b); }
    @compute @workgroup_size(1) fn run() { var x = 1u; forward(&x, &x); }
    """,
    """
    var<private> value = 3u;
    fn readGlobal() -> u32 { return value; }
    fn update(p: ptr<private, u32>) { *p = readGlobal(); }
    @compute @workgroup_size(1) fn run() { update(&value); }
    """,
    """
    @compute @workgroup_size(1) fn run() { var v = vec2u(1u, 2u); let p = &v.x; _ = *p; }
    """,
])
func rejectsInvalidShaderPointerUses(source: String) {
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func shaderPointerParametersMutateOriginalValuesOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(pointerShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        ])
        #expect(try await ticket.wait() == [[0, 0, 30, 219, 10, 11, 35, 219, 20, 21, 35, 219, 0, 0, 30, 219]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
