import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let uniformShader = """
    struct Config {
        normal: vec3u,
        bias: u32,
        @align(32) table: array<vec4u, 2>,
    }
    @group(0) @binding(0) var<uniform> cfg: Config;
    @group(3) @binding(9) var<uniform> tuning: u32;
    @group(0) @binding(1) var<storage, read_write> output: array<u32, 4>;
    fn score(i: u32) -> u32 {
        let row = cfg.table[i];
        return cfg.normal.z + cfg.bias + tuning + row.x + row.w;
    }
    @compute @workgroup_size(4)
    fn run(@builtin(global_invocation_id) id: vec3u) { output[id.x] = score(id.x); }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsComputeUniformBuffersWithWGSLLayout(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(uniformShader), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [16])
    #expect(shader.uniformBindings.map(\.minimumByteSize) == [64, 4])
    #expect(shader.uniformBindings.map(\.binding) == [.init(group: 0, binding: 0), .init(group: 3, binding: 9)])
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .msl:
        #expect(shader.uniformBindings.map(\.flatBindingIndex) == [9, 10])
        #expect(source.contains("constant uint* saga_buffer_0 [[buffer(9)]]"))
    case .hlsl:
        #expect(source.contains("register(b0, space0)") && source.contains("register(b9, space3)"))
        #expect(source.contains("uint4 saga_buffer_0[4]"))
    case .glsl, .wgsl:
        let input: ShaderInput = format == .wgsl ? .wgsl(source) : .glsl(source, stage: .compute, defines: [:])
        let module = try compiler.parseAndValidate(input).module
        let globals = module.globalVariables.values.filter { $0.space == .uniform }
        #expect(globals.count == 2)
        let first = try #require(globals.first)
        guard case .structure(let fields, _) = try module.types.value(for: first.type).kind else {
            Issue.record("Expected uniform structure"); return
        }
        #expect(fields.map(\.offset) == [0, 12, 32])
    default: Issue.record("Unexpected target")
    }
}

@Test func validatesUniformBindingSizeLimitWithoutStorageAlignmentRule() throws {
    let buffer = ResourceID(device: 1, serial: 1)
    let descriptor = BufferDescriptor(size: 65_537, usage: .uniform)
    #expect(try ComputeBufferBinding(buffer: buffer, size: 5).range(in: descriptor, usage: .uniform) == 0..<5)
    #expect(try ComputeBufferBinding(buffer: buffer, size: 65_536).range(in: descriptor, usage: .uniform) == 0..<65_536)
    #expect(throws: GPUError.self) { try ComputeBufferBinding(buffer: buffer).range(in: descriptor, usage: .uniform) }
}

#if canImport(Metal)
@Test func dispatchReadsUniformSlicesAndRejectsInvalidBindingsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(uniformShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let cfg = try await device.createBuffer(.init(size: 320, usage: [.uniform, .copyDestination]))
        let tuning = try await device.createBuffer(.init(size: 4, usage: [.uniform, .copyDestination]))
        let wrongUsage = try await device.createBuffer(.init(size: 64, usage: .storage))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        func dispatch(_ input: ComputeBufferBinding) -> BufferCommand {
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0): input,
                .init(group: 3, binding: 9): .init(buffer: tuning),
                .init(binding: 1): .init(buffer: output),
            ], workgroups: .init(x: 1))
        }
        for invalid in [
            ComputeBufferBinding(buffer: wrongUsage),
            .init(buffer: cfg, offset: 4, size: 64),
            .init(buffer: cfg, offset: 256, size: 60),
        ] {
            await #expect(throws: GPUError.self) { try await device.submitCommands([dispatch(invalid)]) }
        }
        let ticket = try await device.submitCommands([
            .write(buffer: cfg, offset: 256, words: [1, 2, 3, 4, 701, 702, 703, 704, 5, 6, 7, 8, 9, 10, 11, 12]),
            .write(buffer: tuning, offset: 0, words: [10]),
            dispatch(.init(buffer: cfg, offset: 256, size: 64)),
            .copy(source: output, destination: read, byteCount: 16),
            .read(buffer: read, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[30, 38, 17, 17]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
