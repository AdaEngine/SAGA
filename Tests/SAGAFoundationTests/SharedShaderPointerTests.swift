import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let sharedPointerShader = """
    requires unrestricted_pointer_parameters;

    struct Config { values: vec4u, }
    @group(0) @binding(0) var<uniform> config: Config;
    @group(0) @binding(1) var<storage, read> input: array<u32>;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 8>;
    var<workgroup> totals: array<atomic<u32>, 2>;
    var<workgroup> counts: array<atomic<u32>, 2>;

    fn sample(values: ptr<uniform, vec4u>, source: ptr<storage, array<u32>, read>, lane: u32) -> u32 {
        return (*values)[lane] + (*source)[lane] + arrayLength(source);
    }

    fn accumulate(cell: ptr<workgroup, atomic<u32>>, value: u32) -> u32 {
        atomicAdd(cell, value);
        workgroupBarrier();
        return atomicLoad(cell);
    }

    fn relay(cell: ptr<workgroup, atomic<u32>>, value: u32) -> u32 {
        return accumulate(cell, value);
    }

    fn sumAt(cells: ptr<workgroup, array<atomic<u32>, 2>>, index: u32, value: u32) -> u32 {
        return relay(&(*cells)[index], value);
    }

    fn snapshot(cell: ptr<workgroup, atomic<u32>>) -> u32 {
        return workgroupUniformLoad(cell);
    }

    fn publish(destination: ptr<storage, vec4u, read_write>, value: vec4u) {
        *destination = value;
    }

    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
        let value = sample(&config.values, &input, lane);
        let total = sumAt(&totals, group.x, value);
        let count = sumAt(&counts, 1u - group.x, lane + 1u);
        let invalid = sumAt(&totals, group.x + 2u, 999u);
        let synchronized = snapshot(&totals[group.x]);
        let index = group.x * 4u + lane;
        publish(&output[index], vec4u(value, total, count, lane + invalid + synchronized));
        publish(&output[index + 8u], vec4u(999u));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesSharedShaderPointerArguments(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(sharedPointerShader), to: format)
    #expect(shader.storageBindings.map(\.binding.binding) == [1, 2])
    #expect(shader.uniformBindings.map(\.binding.binding) == [0])
    #expect(shader.workgroupStorageByteSize == 32)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl {
        #expect(!generated.contains("requires unrestricted_pointer_parameters"))
        _ = try compiler.parseAndValidate(.wgsl(generated))
    }
}

@Test func rejectsNonuniformSharedPointerArgumentToUniformLoad() throws {
    let source = ShaderInput.wgsl("""
        requires unrestricted_pointer_parameters;
        var<workgroup> values: array<u32, 4>;
        fn snapshot(p: ptr<workgroup, u32>) -> u32 { return workgroupUniformLoad(p); }
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            _ = snapshot(&values[lane]);
        }
        """)
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source)
        Issue.record("Nonuniform pointer argument was accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .nonUniformControlFlow })
        #expect(diagnostics.contains { $0.message.contains("call") })
    }
}

#if canImport(Metal)
@Test func sharedShaderPointersPreserveSynchronizationOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(sharedPointerShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let uniform = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: uniform, offset: 0, words: [10, 20, 30, 40]),
            .write(buffer: input, offset: 0, words: [1, 2, 3, 4]),
            .dispatchCompute(pipeline: pipeline,
                             bindings: [.init(binding: 0): uniform, .init(binding: 1): input, .init(binding: 2): output],
                             workgroups: .init(x: 2)),
            .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
        ])
        let group: [UInt32] = [15, 126, 10, 126, 26, 126, 10, 127, 37, 126, 10, 128, 48, 126, 10, 129]
        #expect(try await ticket.wait() == [group + group])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
