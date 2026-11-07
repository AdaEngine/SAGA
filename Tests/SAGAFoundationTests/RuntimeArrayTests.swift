import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let runtimeArrayShader = """
    @group(0) @binding(1) var<storage, read> src: array<u32>;
    @group(0) @binding(3) var<storage, read_write> dst: array<u32>;
    fn measure() -> u32 { return arrayLength(&src) * 100u + arrayLength(&dst) * 10u; }
    fn read(index: i32) -> u32 { return src[index]; }
    fn write(index: u32, value: u32) { dst[index] = value; }
    fn writeMeasured(index: u32) { write(index, measure() + read(i32(index) - 1i)); }
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        writeMeasured(id.x);
    }
    """

private let storageTailShader = """
    struct Input {
        @size(12) bias: u32,
        factors: array<u32, 2>,
        @align(32) values: array<u32>,
    }
    struct Output {
        count: u32,
        @align(16) values: array<u32>,
    }
    @group(0) @binding(1) var<storage, read> src: Input;
    @group(0) @binding(3) var<storage, read_write> dst: Output;
    fn measure() -> u32 { return arrayLength(&src.values) * 100u + arrayLength(&dst.values) * 10u + src.bias; }
    fn calculate(i: u32) -> u32 { return measure() + src.factors[i] + src.values[i32(i) - 1i]; }
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        if id.x == 0u { dst.count = measure(); }
        dst.values[id.x] = calculate(id.x);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func preservesStorageTailLayoutAndReflection(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let compiled = try compiler.compileCompute(.wgsl(storageTailShader), to: format)
    #expect(compiled.storageBindings.map(\.minimumByteSize) == [64, 32])
    #expect(compiled.storageBindings.map(\.runtimeArrayOffset) == [32, 16])
    #expect(compiled.storageBindings.map(\.runtimeArrayStride) == [4, 4])
    guard case .text(_, let source) = compiled.artifact else { Issue.record("Expected text shader"); return }
    switch format {
    case .msl:
        #expect(source.contains("saga_lengths [[buffer(8)]]"))
        #expect(source.contains("+ 8u"))
    case .hlsl:
        #expect(source.contains("saga_buffer_0.GetDimensions(") && source.contains("count - 8u"))
        #expect(source.contains("saga_buffer_1.GetDimensions(") && source.contains("count - 4u"))
    case .glsl, .wgsl:
        let input: ShaderInput = format == .glsl ? .glsl(source, stage: .compute, defines: [:]) : .wgsl(source)
        let module = try compiler.parseAndValidate(input).module
        let memberOffsets = try module.globalVariables.values.filter { $0.binding != nil }.map { global in
            guard case .structure(let members, _) = try module.types.value(for: global.type).kind else {
                throw ShaderFailure.unsupported("Expected storage structure")
            }
            return members.map(\.offset)
        }
        #expect(memberOffsets == [[0, 12, 32], [0, 16]])
    default: Issue.record("Unexpected target")
    }
}

@Test(arguments: [2, 8])
func rejectsMisalignedAndOverlappingGLSLStorageOffsets(offset: Int) {
    let shader = """
        #version 450 core
        layout(local_size_x = 1) in;
        layout(std430, binding = 0) buffer Data { uvec4 header; layout(offset = \(offset)) uint values[]; } data;
        void main() { data.values[0] = data.header.x; }
        """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.glsl(shader, stage: .compute, defines: [:]))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsRuntimeArrayLengthsAndBoundsForEachTarget(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(runtimeArrayShader), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [4, 4])
    #expect(shader.storageBindings.map(\.runtimeArrayStride) == [4, 4])
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected text shader"); return }
    switch format {
    case .msl:
        #expect(shader.metalRuntimeLengthBuffer == 8)
        #expect(source.contains("saga_lengths [[buffer(8)]]"))
        #expect(source.contains("saga_lengths[0]") && source.contains("saga_lengths[1]"))
    case .hlsl:
        #expect(source.contains("saga_buffer_0.GetDimensions(") && source.contains("saga_buffer_1.GetDimensions("))
    case .glsl:
        #expect(source.contains("uint value[]") && source.contains(".value.length()"))
        _ = try compiler.parseAndValidate(.glsl(source, stage: .compute, defines: [:]))
    case .wgsl:
        #expect(source.contains("arrayLength(&"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target")
    }
}

@Test func arrayLengthRequiresAddressOfRuntimeStorageArray() {
    let compiler = IndependentShaderCompiler()
    for (declaration, value) in [
        ("@group(0) @binding(0) var<storage, read> values: array<u32, 4>;", "&values"),
        ("@group(0) @binding(0) var<storage, read> values: array<u32>;", "values"),
        ("@group(0) @binding(0) var<storage, read> values: array<u32>;", "&values[0u]"),
    ] {
        let source = declaration + "\n@compute @workgroup_size(1) fn run() { _ = arrayLength(\(value)); }"
        #expect(throws: ShaderFailure.self) { _ = try compiler.parseAndValidate(.wgsl(source)) }
    }
}

@Test func validatesStorageRangesAndUsageCompatibility() throws {
    let buffer = ResourceID(device: 1, serial: 1)
    let descriptors = [buffer: BufferDescriptor(size: 512, usage: .storage)]
    let read = ComputeBindingSlot(binding: 1)
    let write = ComputeBindingSlot(binding: 3)
    let pipeline = ComputePipeline(id: .init(device: 1, serial: 2), storageBindings: [
        .init(slot: read, minimumByteSize: 4, isReadOnly: true),
        .init(slot: write, minimumByteSize: 4, isReadOnly: false),
    ])
    func command(_ source: StorageBufferBinding, _ destination: StorageBufferBinding,
                 with layout: ComputePipeline = pipeline) -> BufferCommand {
        .dispatchComputeRanges(pipeline: layout, bindings: [read: source, write: destination], workgroups: .init(x: 1))
    }
    let first = StorageBufferBinding(buffer: buffer, size: 12)
    let second = StorageBufferBinding(buffer: buffer, offset: 256, size: 20)
    #expect(throws: GPUError.self) { try command(first, second).validate(device: 1, descriptors: descriptors) }
    for readOnly in [true, false] {
        let sameAccess = ComputePipeline(id: pipeline.id, storageBindings: [
            .init(slot: read, minimumByteSize: 4, isReadOnly: readOnly),
            .init(slot: write, minimumByteSize: 4, isReadOnly: readOnly),
        ])
        try command(first, second, with: sameAccess).validate(device: 1, descriptors: descriptors)
        if readOnly {
            try command(first, first, with: sameAccess).validate(device: 1, descriptors: descriptors)
        } else {
            #expect(throws: GPUError.self) {
                try command(first, first, with: sameAccess).validate(device: 1, descriptors: descriptors)
            }
        }
    }
    let output = ResourceID(device: 1, serial: 3)
    let separate = [buffer: descriptors[buffer]!, output: descriptors[buffer]!]
    try command(first, .init(buffer: output, offset: 256, size: 20)).validate(device: 1, descriptors: separate)
    for invalid in [
        StorageBufferBinding(buffer: output, offset: 4, size: 20),
        .init(buffer: output, offset: 256, size: 2),
        .init(buffer: output, offset: 256, size: 260),
        .init(buffer: output, offset: UInt64.max - 255),
    ] {
        #expect(throws: GPUError.self) { try command(first, invalid).validate(device: 1, descriptors: separate) }
    }
}

#if canImport(Metal)
@Test func storageTailUsesBoundSizeAndMemberOffsetOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(storageTailShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 324, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 296, usage: [.storage, .copyDestination, .copySource]))
        let read = try await device.createBuffer(.init(size: 296, usage: [.copyDestination, .mapRead]))
        let bindings: [ComputeBindingSlot: StorageBufferBinding] = [
            .init(binding: 1): .init(buffer: input, offset: 256, size: 64),
            .init(binding: 3): .init(buffer: output, offset: 256, size: 36),
        ]
        var tooSmall = bindings
        tooSmall[.init(binding: 3)] = .init(buffer: output, offset: 256, size: 28)
        var minimum = bindings
        minimum[.init(binding: 3)] = .init(buffer: output, offset: 256, size: 32)
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([
                .dispatchComputeRanges(pipeline: pipeline, bindings: tooSmall, workgroups: .init(x: 1))
            ])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 256, words: [7, 888, 888, 10, 20, 888, 888, 888, 1, 2, 3, 0, 0, 0, 0, 0, 777]),
            .write(buffer: output, offset: 256, words: [0, 888, 888, 888, 0, 0, 0, 0, 0, 999]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 6)),
            .copy(source: output, destination: read, byteCount: 296),
            .read(buffer: read, byteCount: 296),
            .dispatchComputeRanges(pipeline: pipeline, bindings: minimum, workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 296),
            .read(buffer: read, byteCount: 296),
        ])
        let prefix = Array(repeating: UInt32(0), count: 64)
        let words = try await ticket.wait()
        #expect(words == [
            prefix + [857, 888, 888, 888, 867, 878, 859, 860, 857, 999],
            prefix + [847, 888, 888, 888, 857, 878, 859, 860, 857, 999],
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func runtimeArrayLengthUsesBoundRangesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(runtimeArrayShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 272, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 280, usage: [.storage, .copyDestination, .copySource]))
        let read = try await device.createBuffer(.init(size: 280, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 256, words: [1, 2, 3, 777]),
            .write(buffer: output, offset: 256, words: [0, 0, 0, 0, 0, 999]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 1): .init(buffer: input, offset: 256, size: 12),
                .init(binding: 3): .init(buffer: output, offset: 256, size: 20),
            ], workgroups: .init(x: 6)),
            .copy(source: output, destination: read, byteCount: 280),
            .read(buffer: read, byteCount: 280),
        ])
        let snapshots = try await ticket.wait()
        let words = try #require(snapshots.first)
        #expect(words.count == 70)
        #expect(words.prefix(64).allSatisfy { $0 == 0 })
        #expect(Array(words.suffix(6)) == [350, 351, 352, 353, 350, 999])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
