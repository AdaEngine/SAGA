import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvUniformInput = ShaderInput.wgsl("""
    struct Entry { data: vec4u }
    struct Cell { entries: array<Entry, 2>, transform: mat2x3f }
    struct Settings { flags: u32, @align(16) cells: array<Cell, 2>, tail: vec4u }
    @group(1) @binding(3) var<uniform> settings: Settings;
    @group(2) @binding(0) var<uniform> tuning: u32;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 4>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
        let cell = settings.cells[lane];
        let entry = cell.entries[1];
        let transformed = cell.transform * vec2f(2.0, 3.0);
        output[lane] = vec4u(entry.data.x + settings.flags + tuning, entry.data.w + settings.tail.x,
                             bitcast<u32>(transformed.x), bitcast<u32>(transformed.z));
        let signedEntry = cell.entries[i32(lane) - 1i];
        let unsignedEntry = cell.entries[lane + 1u];
        output[lane + 2u] = vec4u(signedEntry.data.x, unsignedEntry.data.x, signedEntry.data.w, unsignedEntry.data.w);
    }
    """)

private func exportUniforms(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvUniformInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvUniformCodeGeneratorRoundTripsStd140UniformBlocks(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportUniforms(compiler)
    try #require(words.count > 5)
    var uniformPointers: [UInt32: UInt32] = [:]
    var uniformVariables: [UInt32] = []
    var blocks: Set<UInt32> = []
    var strides: Set<UInt32> = []
    var memberOffsets: [UInt32: Set<UInt32>] = [:]
    var hasMatrixStride16 = false
    var hasColumnMajor = false
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 32 where count == 4 && words[offset + 2] == 2: uniformPointers[words[offset + 1]] = words[offset + 3]
        case 59 where count >= 4 && words[offset + 3] == 2: uniformVariables.append(words[offset + 1])
        case 71 where count == 3 && words[offset + 2] == 2: blocks.insert(words[offset + 1])
        case 71 where count == 4 && words[offset + 2] == 6: strides.insert(words[offset + 3])
        case 72 where count == 5 && words[offset + 3] == 35: memberOffsets[words[offset + 1], default: []].insert(words[offset + 4])
        case 72 where count == 5 && words[offset + 3] == 7: hasMatrixStride16 = hasMatrixStride16 || words[offset + 4] == 16
        case 72 where count == 4 && words[offset + 3] == 5: hasColumnMajor = true
        default: break
        }
        offset += count
    }
    #expect(uniformVariables.count == 2)
    #expect(uniformVariables.allSatisfy { pointer in uniformPointers[pointer].map { blocks.contains($0) } ?? false })
    #expect(strides.isSuperset(of: [16, 64]))
    #expect(memberOffsets.values.contains([0, 16, 144]))
    #expect(hasMatrixStride16 && hasColumnMajor)
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.uniformBindings.map(\.binding) == [IRResourceBinding(group: 1, binding: 3), .init(group: 2, binding: 0)])
    #expect(shader.uniformBindings.first?.minimumByteSize == 160)
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

@Test(arguments: [false, true])
func spirvUniformCodeGeneratorDistinguishesPackedMatricesFromWGSLStandardArrayLayout(_ packedArray: Bool) throws {
    let source: ShaderInput = if packedArray {
        .wgsl("""
            requires uniform_buffer_standard_layout;
            @group(0) @binding(0) var<uniform> values: array<u32, 4>;
            @group(0) @binding(1) var<storage, read_write> output: vec4u;
            @compute @workgroup_size(1) fn run() { output = vec4u(values[0], values[1], values[2], values[3]); }
            """)
    } else {
        .wgsl("""
            @group(0) @binding(0) var<uniform> scale: mat2x2f;
            @group(0) @binding(1) var<storage, read_write> output: vec2f;
            @compute @workgroup_size(1) fn run() { output = scale * vec2f(2.0, 3.0); }
            """)
    }
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    try #require(words.count > 5)
    var hasPackedStride = false
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        if packedArray && opcode == 71 && count == 4 && words[offset + 2] == 6 && words[offset + 3] == 4 {
            hasPackedStride = true // ArrayStride 4
        }
        if !packedArray && opcode == 72 && count == 5 && words[offset + 3] == 7 && words[offset + 4] == 8 {
            hasPackedStride = true // MatrixStride 8
        }
        offset += count
    }
    #expect(hasPackedStride)
    let input = ShaderInput.spirv(words)
    let raw = try compiler.parseAndValidate(input).module
    #expect(raw.uniformBufferStandardLayout == packedArray)
    let metal = try compiler.compileCompute(input, to: .msl)
    #expect(metal.uniformBindings.map(\.minimumByteSize) == [16])
    let wgsl = try compiler.compileCompute(input, to: .wgsl)
    #expect(wgsl.requiresUniformBufferStandardLayout == packedArray)
    guard case .text(.wgsl, let text) = wgsl.artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.wgsl(text))
}

#if canImport(Metal)
@Test func spirvUniformCodeGeneratorReadsNestedUniformAggregatesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportUniforms(compiler)), to: .msl)
    var uniforms: [UInt32] = [3, 901, 902, 903, 1, 2, 3, 4, 5, 6, 7, 8]
    uniforms += [Float(1), 2, 3, -99, 4, 5, 6, -99].map(\.bitPattern)
    uniforms += [10, 11, 12, 13, 14, 15, 16, 17]
    uniforms += [Float(7), 8, 9, -99, 10, 11, 12, -99].map(\.bitPattern)
    uniforms += [21, 904, 905, 906]
    let expected: [UInt32] = [17, 29, Float(14).bitPattern, Float(24).bitPattern,
                              26, 38, Float(44).bitPattern, Float(54).bitPattern, 0, 5, 0, 8, 10, 0, 13, 0]
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let settings = try await device.createBuffer(.init(size: 416, usage: [.uniform, .copyDestination]))
        let tuning = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: settings, offset: 256, words: uniforms),
            .write(buffer: tuning, offset: 0, words: [9, 0, 0, 0]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(group: 1, binding: 3): .init(buffer: settings, offset: 256, size: 160),
                .init(group: 2, binding: 0): .init(buffer: tuning),
                .init(binding: 2): .init(buffer: output),
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        ]).wait()
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
