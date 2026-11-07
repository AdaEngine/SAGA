import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let groupedStorageShader = """
    @group(3) @binding(999) var<storage, read> src: array<u32>;
    @group(0) @binding(5) var<storage, read> gain: u32;
    @group(1) @binding(999) var<storage, read_write> dst: array<u32>;
    fn scaled(i: u32) -> u32 { return src[i] * gain + arrayLength(&src) * 10u + arrayLength(&dst); }
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) { dst[id.x] = scaled(id.x); }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func preservesLogicalStorageBindingsAcrossGroups(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(groupedStorageShader), to: format)
    let logical: [IRResourceBinding] = [
        .init(group: 3, binding: 999), .init(group: 0, binding: 5), .init(group: 1, binding: 999),
    ]
    #expect(shader.storageBindings.map(\.binding) == logical)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected text shader"); return }
    switch format {
    case .msl:
        #expect(shader.storageBindings.map(\.flatBindingIndex) == [1, 5, 0])
        #expect(source.contains("saga_buffer_0 [[buffer(1)]]"))
        #expect(source.contains("saga_buffer_1 [[buffer(5)]]"))
        #expect(source.contains("saga_buffer_2 [[buffer(0)]]"))
        #expect(shader.metalRuntimeLengthBuffer == 8)
    case .hlsl:
        #expect(shader.storageBindings.allSatisfy { $0.flatBindingIndex == nil })
        #expect(source.contains("register(t999, space3)"))
        #expect(source.contains("register(t5, space0)"))
        #expect(source.contains("register(u999, space1)"))
    case .glsl:
        #expect(shader.storageBindings.map(\.flatBindingIndex) == [1, 5, 0])
        let module = try compiler.parseAndValidate(.glsl(source, stage: .compute, defines: [:])).module
        #expect(module.globalVariables.values.compactMap(\.binding).map(\.binding) == [1, 5, 0])
    case .wgsl:
        #expect(shader.storageBindings.allSatisfy { $0.flatBindingIndex == nil })
        let module = try compiler.parseAndValidate(.wgsl(source)).module
        #expect(module.globalVariables.values.compactMap(\.binding) == logical)
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    "@group(4) @binding(0) var<storage, read> data: u32;",
    "@group(0) @binding(1000) var<storage, read> data: u32;",
    (0..<9).map { "@group(\($0 % 4)) @binding(\($0 / 4)) var<storage, read> data\($0): u32;" }
        .joined(separator: "\n"),
])
func rejectsComputeStorageLayoutsOutsidePortableLimits(declarations: String) {
    let source = declarations + "\n@compute @workgroup_size(1) fn run() {}"
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl) }
}

#if canImport(Metal)
@Test func metalRejectsPipelineWithUnrelatedStorageLayout() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let otherBackend = try MetalBackend.makeDevice()
    do {
        let compiler = IndependentShaderCompiler()
        let shader = try compiler.compileCompute(.wgsl(groupedStorageShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let unrelated = try compiler.compileCompute(.wgsl("""
            @group(1) @binding(999) var<storage, read_write> dst: array<u32, 2>;
            @compute @workgroup_size(1) fn run() { dst[1] = 1u; }
            """), to: .msl)
        await #expect(throws: GPUError.validation("Metal compute pipeline layout does not match compiled shader")) {
            try await otherBackend.allocateComputePipeline(pipeline, shader: unrelated)
        }
    } catch {
        await otherBackend.close(owner: device.identity)
        await device.close()
        throw error
    }
    await otherBackend.close(owner: device.identity)
    await device.close()
}

@Test func dispatchesSparseStorageGroupsAndRuntimeLengthsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(groupedStorageShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 268, usage: [.storage, .copyDestination]))
        let gain = try await device.createBuffer(.init(size: 4, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 272, usage: [.storage, .copyDestination, .copySource]))
        let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
        let inputSlot = ComputeBindingSlot(group: 3, binding: 999)
        let bindings: [ComputeBindingSlot: StorageBufferBinding] = [
            inputSlot: .init(buffer: input, offset: 256, size: 8),
            .init(binding: 5): .init(buffer: gain),
            .init(group: 1, binding: 999): .init(buffer: output, offset: 256, size: 12),
        ]
        var wrongGroup = bindings
        wrongGroup[.init(group: 2, binding: 999)] = wrongGroup.removeValue(forKey: inputSlot)
        await #expect(throws: GPUError.validation("Compute bindings do not match the pipeline layout")) {
            try await device.submitCommands([
                .dispatchComputeRanges(pipeline: pipeline, bindings: wrongGroup, workgroups: .init(x: 1)),
            ])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 256, words: [3, 5, 777]),
            .write(buffer: gain, offset: 0, words: [2]),
            .write(buffer: output, offset: 256, words: [0, 0, 0, 999]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 4)),
            .copy(source: output, destination: read, byteCount: 272),
            .read(buffer: read, byteCount: 272),
        ])
        #expect(try await ticket.wait() == [Array(repeating: 0, count: 64) + [29, 33, 23, 999]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
