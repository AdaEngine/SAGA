import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let workgroupAggregateShader = """
    struct Slot { bins: array<vec2u, 2>, stamp: u32, }
    struct Counters { hits: atomic<u32>, parity: array<atomic<u32>, 2>, }
    override count = 4u;
    var<workgroup> slots: array<Slot, count>;
    var<workgroup> counters: Counters;
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 8>;

    fn fill(lane: u32, base: u32) {
        slots[lane].bins[1] = vec2u(base + lane + 1u, 2u);
        slots[lane].stamp = 31u + lane;
        atomicAdd(&counters.hits, 1u);
        atomicAdd(&counters.parity[lane % 2u], 1u);
    }

    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
        let initial = slots[lane].bins[1].x + slots[lane].stamp
                    + atomicLoad(&counters.hits) + atomicLoad(&counters.parity[lane % 2u]);
        workgroupBarrier();
        fill(lane, group.x * 10u);
        workgroupBarrier();
        let next = slots[(lane + 1u) % 4u];
        let shared = workgroupUniformLoad(&slots[group.x]);
        let missing = workgroupUniformLoad(&slots[count + group.x]);
        let uniformBin = workgroupUniformLoad(&slots[group.x].bins[group.x + 1u]);
        if shared.stamp != 0u { workgroupBarrier(); }
        let invalid = lane + 2u;
        slots[lane].bins[invalid].x = 999u;
        output[group.x * 4u + lane] = vec4u(next.bins[1].x + initial * 1000u, slots[lane].stamp,
                                          atomicLoad(&counters.hits),
                                          atomicLoad(&counters.parity[lane % 2u]) + slots[lane].bins[invalid].x
                                              + shared.stamp + missing.stamp + uniformBin.y);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesNestedWorkgroupAggregates(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(workgroupAggregateShader)
    let defaults = try compiler.compileCompute(source, to: format)
    let specialized = try compiler.compileCompute(source, to: format, constants: ["count": 5])
    #expect(defaults.workgroupStorageByteSize == 112)
    #expect(specialized.workgroupStorageByteSize == 144)
    guard case .text(_, let generated) = specialized.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
}

@Test func reflectsSpecializedWorkgroupAggregateStorageAboveTheDefaultLimit() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(workgroupAggregateShader)
    let shader = try compiler.compileCompute(source, to: .msl, constants: ["count": 700])
    #expect(shader.workgroupStorageByteSize == 16_816)
}

#if canImport(Metal)
@Test func readsAndWritesNestedWorkgroupAggregatesOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(workgroupAggregateShader), to: .msl,
                                                                 constants: ["count": 5])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 2)),
            .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
        ])
        #expect(try await ticket.wait() == [[
            2, 31, 4, 35, 3, 32, 4, 35, 4, 33, 4, 35, 1, 34, 4, 35,
            12, 31, 4, 34, 13, 32, 4, 34, 14, 33, 4, 34, 11, 34, 4, 34,
        ]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
