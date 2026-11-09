import Testing
import SAGACompiler
import SAGAShaderIR
import SAGACore
import SAGATypes
import SAGAMetal

private func bufferInt64AtomicSource(signed: Bool, nested: Bool = false) -> String {
    let minimumType = signed ? "i64" : "u64"
    let minimum = signed ? "-9223372036854775808li" : "9223372036854775808lu"
    let atomicFields = """
        largest: atomic<u64>,
        smallest: atomic<\(minimumType)>,
    """
    let nestedType = nested ? "struct Extrema { \(atomicFields) }" : ""
    let fields = nested ? "extrema: Extrema," : atomicFields
    let extrema = nested ? "values.extrema" : "values"
    return """
    \(nestedType)
    struct Values {
        \(fields)
        tail: array<atomic<u64>>,
    }
    @group(0) @binding(0) var<storage, read_write> values: Values;
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        atomicMax(&\(extrema).largest, 0xfedcba9876543210lu);
        atomicMin(&\(extrema).smallest, \(minimum));
        atomicMax(&values.tail[lane], 0x9000000000000001lu);
    }
    """
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func bufferInt64AtomicMinMaxPreservesOffsetsAndWidth(format: ShaderOutputFormat, roundTrip: Bool) throws {
    let compiler = IndependentShaderCompiler()
    var input = ShaderInput.wgsl(bufferInt64AtomicSource(signed: format != .msl))
    if roundTrip {
        guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
            throw ShaderFailure.incompatibleArtifact
        }
        input = .spirv(words)
    }
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.requiresShaderInt64AtomicMinMax && !shader.requiresShaderInt64AtomicAllOps)
    let binding = try #require(shader.storageBindings.first)
    #expect(binding.minimumByteSize == 24)
    #expect(binding.runtimeArrayOffset == 16)
    #expect(binding.runtimeArrayStride == 8)
    guard case .text(_, let text) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    #expect(text.contains("18364758544493064720"))
    switch format {
    case .wgsl:
        #expect(text.contains("atomic<i64>"))
        _ = try compiler.parseAndValidate(.wgsl(text),
            capabilities: .referenceDefault.union([.shaderInt64, .shaderInt64AtomicMinMax]))
    case .msl:
        #expect(text.contains("atomic_min_explicit"))
        #expect(text.contains("atomic_ulong"))
        #expect(throws: ShaderFailure.unsupported("Metal storage min/max atomics require unsigned 64-bit values")) {
            try compiler.compileCompute(.wgsl(bufferInt64AtomicSource(signed: true)), to: .msl)
        }
    case .hlsl:
        #expect(text.contains("InterlockedMin64"))
        #expect(shader.hlslProfile == "cs_6_6")
    case .glsl:
        #expect(text.contains("atomicMin("))
        #expect(text.contains("GL_EXT_shader_atomic_int64") || text.contains("GL_NV_shader_atomic_int64"))
    default: Issue.record("Unexpected output format")
    }
}

private func glslNestedInt64AtomicSource(fragment: Bool) -> String {
    let entry = fragment ? "@fragment fn run(@builtin(position) position: vec4f)" :
        "@compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32)"
    let lane = fragment ? "let lane = u32(position.x);" : ""
    return """
    enable f16;
    enable wgpu_binding_array;
    struct Cell {
        marker: f16,
        @align(32) maximum: atomic<u64>,
        minimum: atomic<i64>,
        legacy: atomic<i32>,
    }
    struct Values {
        header: u32,
        grid: array<array<Cell, 2>, 2>,
        tail: array<Cell>,
    }
    @group(0) @binding(0) var<storage, read_write> values: Values;
    @group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
    @group(0) @binding(2) var<storage, read_write> slots: binding_array<Values, 2>;
    \(entry) {
        \(lane)
        values.grid[1][lane].marker = 1.5h;
        atomicMax(&values.grid[1][lane].maximum, 0xfedcba9876543210lu);
        atomicMin(&values.tail[lane].minimum, -9223372036854775808li);
        atomicMin(&cells[lane].legacy, -3i);
        atomicMax(&slots[lane].tail[lane].maximum, 0x9000000000000001lu);
        values.header = arrayLength(&slots[lane].tail);
    }
    """
}

@Test(arguments: [false, true])
func bufferInt64AtomicGLSLKeepsNestedLayoutAndTypedLValues(fragment: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(glslNestedInt64AtomicSource(fragment: fragment))
    let bindings: [ComputeShaderArtifact.StorageBinding]
    let artifact: ShaderArtifact
    if fragment {
        let shader = try compiler.compileRenderStage(input, entryPoint: "run", to: .glsl)
        bindings = shader.storageBindings
        artifact = shader.artifact
    } else {
        let shader = try compiler.compileCompute(input, to: .glsl)
        bindings = shader.storageBindings
        artifact = shader.artifact
    }
    #expect(bindings.map(\.minimumByteSize) == [352, 64, 352])
    #expect(bindings.map(\.runtimeArrayOffset) == [288, 0, 288])
    #expect(bindings.map(\.runtimeArrayStride) == [64, 64, 64])
    #expect(bindings.last?.bindingArray?.count == 2)
    guard case .text(_, let text) = artifact else {
        Issue.record("Expected GLSL")
        return
    }
    #expect(text.contains("uint64_t saga_member_1;"))
    #expect(text.contains("int64_t saga_member_2;"))
    #expect(text.contains("saga_member_1[2][2];"))
    #expect(text.contains("GL_EXT_shader_16bit_storage"))
    #expect(text.contains("atomicMin(saga_buffer_1.saga_data["))
    #expect(text.contains("atomicMax(saga_buffer_2_0.saga_member_2["))
    #expect(text.contains("atomicMax(saga_buffer_2_1.saga_member_2["))
    #expect(text.contains("saga_buffer_2_1.saga_member_2.length()"))
}

@Test(arguments: [
    "let old = atomicMax(&value, 1lu);",
    "atomicAdd(&value, 1lu);",
    "let old = atomicLoad(&value);",
    "atomicStore(&value, 1lu);",
])
func bufferInt64AtomicMinMaxTierRejectsOtherOperations(_ operation: String) {
    let compiler = IndependentShaderCompiler()
    let source = """
        @group(0) @binding(0) var<storage, read_write> value: atomic<u64>;
        @compute @workgroup_size(1) fn run() { \(operation) }
        """
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl(source),
            capabilities: .referenceDefault.union([.shaderInt64, .shaderInt64AtomicMinMax]))
    }
}

@Test func bufferInt64AtomicMinMaxTierRejectsWorkgroupUse() {
    let source = """
        var<workgroup> value: atomic<u64>;
        @compute @workgroup_size(1) fn run() { atomicMax(&value, 1lu); }
        """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source),
            capabilities: .referenceDefault.union([.shaderInt64, .shaderInt64AtomicMinMax]))
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func bufferInt64AtomicMinMaxUpdatesWideStorageOnMetal(nested: Bool) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.shaderInt64AtomicMinMax) else { return }
    let source = bufferInt64AtomicSource(signed: false, nested: nested)
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl)
    #expect(shader.requiresShaderInt64AtomicMinMax)
    let pipeline = try await device.createComputePipeline(shader)
    let storage = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource, .copyDestination]))
    let readback = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
    let words = try await device.submitCommands([
        .write(buffer: storage, offset: 0, words: [1, 0x8000_0000, .max, .max, 3, 0, 7, 0]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): storage], workgroups: .init(x: 1)),
        .copy(source: storage, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
    ]).wait()
    #expect(words == [[0x7654_3210, 0xfedc_ba98, 0, 0x8000_0000, 1, 0x9000_0000, 1, 0x9000_0000]])
    await device.close()
}
#endif
