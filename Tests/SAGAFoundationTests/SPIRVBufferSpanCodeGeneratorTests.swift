import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvBufferSpanInput = ShaderInput.wgsl("""
    struct Pair { dims: vec2u, @size(8) tag: u32 }
    struct Cell { entries: array<Pair, 2>, @size(16) tail: u32 }
    struct Settings { @size(16) flags: u32, cells: array<Cell, 2>, @size(16) tail: u32 }
    struct Record { data: vec2u, @size(24) last: u32 }
    struct Store { records: array<Record, 2>, @size(16) tail: u32 }
    @group(0) @binding(0) var<uniform> settings: Settings;
    @group(0) @binding(1) var<storage, read_write> state: Store;
    @group(0) @binding(2) var<storage, read_write> output: vec4u;
    @compute @workgroup_size(1) fn run() {
        let first = settings.cells[0];
        let second = settings.cells[1];
        let picked = state.records[1];
        output = vec4u(first.entries[1].dims.y + first.tail, second.entries[0].tag + settings.tail,
                       picked.data.x, state.tail);
        state.records[1].last = picked.last + settings.flags;
    }
    """)

private func exportBufferSpan(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvBufferSpanInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvBufferSpanCodeGeneratorPreservesReflectedSizesAndArrayStrides(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let direct = try compiler.compileCompute(spirvBufferSpanInput, to: format)
    #expect(direct.uniformBindings.map(\.minimumByteSize) == [128])
    #expect(direct.storageBindings.map(\.minimumByteSize) == [80, 16])

    let words = try exportBufferSpan(compiler)
    try #require(words.count > 5)
    var strides: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        if words[offset] & 0xffff == 71 && count == 4 && words[offset + 2] == 6 {
            strides.insert(words[offset + 3]) // ArrayStride
        }
        offset += count
    }
    #expect(strides.isSuperset(of: [16, 32, 48]))
    let roundTrip = try compiler.compileCompute(.spirv(words), to: format)
    #expect(roundTrip.uniformBindings.map(\.minimumByteSize) == direct.uniformBindings.map(\.minimumByteSize))
    #expect(roundTrip.storageBindings.map(\.minimumByteSize) == direct.storageBindings.map(\.minimumByteSize))
    if format == .wgsl {
        guard case .text(.wgsl, let text) = roundTrip.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

@Test(arguments: [119, 127])
func spirvBufferSpanCodeGeneratorRejectsInvalidDeclaredSpans(_ span: Int) throws {
    let compiler = IndependentShaderCompiler()
    var words = try exportBufferSpan(compiler)
    let prefix = "saga.struct-span.v1:128:"
    var marker: (range: Range<Int>, suffix: Substring)?
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        if words[offset] & 0xffff == 5 && count >= 3 { // OpName
            let range = (offset + 2)..<(offset + count)
            let bytes = words[range].flatMap { word in (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) } }
            let name = String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
            if name.hasPrefix(prefix) {
                marker = (range, name.dropFirst(prefix.count))
                break
            }
        }
        offset += count
    }
    let field = try #require(marker)
    let text = Array("saga.struct-span.v1:\(span):\(field.suffix)".utf8) + [0]
    try #require(text.count <= field.range.count * 4)
    let bytes = text + Array(repeating: 0, count: field.range.count * 4 - text.count)
    words.replaceSubrange(field.range, with: stride(from: 0, to: bytes.count, by: 4).map { index in
        UInt32(bytes[index]) | UInt32(bytes[index + 1]) << 8 | UInt32(bytes[index + 2]) << 16 | UInt32(bytes[index + 3]) << 24
    })
    do {
        _ = try compiler.parseAndValidate(.spirv(words))
        Issue.record("Accepted a smaller or misaligned declared structure span")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .invalidBinary })
    }
}

#if canImport(Metal)
@Test func spirvBufferSpanCodeGeneratorPreservesPaddedBufferAccessOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportBufferSpan(compiler)), to: .msl)
    #expect(shader.uniformBindings.map(\.minimumByteSize) == [128])
    #expect(shader.storageBindings.map(\.minimumByteSize) == [80, 16])
    let uniforms: [UInt32] = [
        3, 901, 902, 903, 1, 2, 3, 904, 4, 5, 6, 905, 7, 906, 907, 908,
        8, 9, 10, 909, 11, 12, 13, 910, 14, 911, 912, 913, 15, 914, 915, 916,
    ]
    let storage: [UInt32] = [
        101, 102, 103, 801, 802, 803, 804, 805, 201, 202, 203, 806, 807, 808, 809, 810,
        301, 811, 812, 813,
    ]
    var expectedStorage = storage
    expectedStorage[10] = 206
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let settings = try await device.createBuffer(.init(size: 128, usage: [.uniform, .copyDestination]))
        let state = try await device.createBuffer(.init(size: 80, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: settings, offset: 0, words: uniforms),
            .write(buffer: state, offset: 0, words: storage),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): settings, .init(binding: 1): state,
                                                             .init(binding: 2): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .copy(source: state, destination: readback, byteCount: 80), .read(buffer: readback, byteCount: 80),
        ]).wait()
        #expect(snapshots == [[12, 25, 201, 301], expectedStorage])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func spirvBufferSpanCodeGeneratorPreservesAlignedRuntimeTailOnMetal() async throws {
    let source = ShaderInput.wgsl("""
        struct Packet { flag: u32, @align(32) values: array<u32> }
        @group(0) @binding(0) var<storage, read_write> packet: Packet;
        @group(0) @binding(1) var<storage, read_write> output: vec4u;
        @compute @workgroup_size(1) fn run() {
            let count = arrayLength(&packet.values);
            let previous = packet.values[7];
            output = vec4u(packet.flag, count, packet.values[0], previous);
            packet.values[7] = previous + packet.flag;
        }
        """)
    let compiler = IndependentShaderCompiler()
    let direct = try compiler.compileCompute(source, to: .msl)
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    for compiled in [direct, shader] {
        let binding = try #require(compiled.storageBindings.first)
        #expect(binding.minimumByteSize == 64)
        #expect(binding.runtimeArrayOffset == 32 && binding.runtimeArrayStride == 4)
    }
    let initial: [UInt32] = [7, 901, 902, 903, 904, 905, 906, 907, 101, 102, 103, 104, 105, 106, 107, 108]
    var expected = initial
    expected[15] = 115
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let packet = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: packet, offset: 0, words: initial),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): packet, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .copy(source: packet, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        ]).wait()
        #expect(snapshots == [[7, 8, 101, 108], expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
