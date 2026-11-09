import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private enum AtomicExportScalar: String, CaseIterable, Sendable {
    case i32, u32
}

private func atomicExportInput(_ scalar: AtomicExportScalar) -> ShaderInput {
    let suffix = scalar == .i32 ? "i" : "u"
    let start = scalar == .i32 ? "-4i" : "0xfffffffcu"
    let word = scalar == .i32 ? "bitcast<u32>" : "u32"
    return .wgsl("""
        @group(0) @binding(0) var<storage, read_write> counter: atomic<\(scalar.rawValue)>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 14>;

        @compute @workgroup_size(1) fn run() {
            atomicStore(&counter, \(start));
            output[0] = \(word)(atomicLoad(&counter));
            output[1] = \(word)(atomicAdd(&counter, 7\(suffix)));
            output[2] = \(word)(atomicSub(&counter, 5\(suffix)));
            output[3] = \(word)(atomicMin(&counter, 2\(suffix)));
            output[4] = \(word)(atomicMax(&counter, 5\(suffix)));
            output[5] = \(word)(atomicAnd(&counter, 6\(suffix)));
            output[6] = \(word)(atomicOr(&counter, 3\(suffix)));
            output[7] = \(word)(atomicXor(&counter, 2\(suffix)));
            output[8] = \(word)(atomicExchange(&counter, 11\(suffix)));
            let matched = atomicCompareExchangeWeak(&counter, 11\(suffix), 17\(suffix));
            output[9] = \(word)(matched.old_value);
            output[10] = select(0u, 1u, matched.exchanged);
            output[11] = \(word)(atomicLoad(&counter));
            let mismatched = atomicCompareExchangeWeak(&counter, 99\(suffix), 42\(suffix));
            output[12] = \(word)(mismatched.old_value);
            output[13] = select(0u, 1u, mismatched.exchanged);
        }
        """)
}

private func exportStorageAtomics(_ scalar: AtomicExportScalar, with compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(atomicExportInput(scalar), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func expectRelaxedAtomicInstructions(_ words: [UInt32], scalar: AtomicExportScalar) throws {
    try #require(words.count > 5)
    var constants: [UInt32: UInt32] = [:]
    var opcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        if opcode == 43 && count == 4 { constants[words[offset + 2]] = words[offset + 3] } // OpConstant
        if (227...242).contains(opcode) {
            opcodes.insert(opcode)
            let scope = offset + (opcode == 228 ? 2 : 4)
            try #require(scope + 1 < offset + count)
            #expect(constants[words[scope]] == 1) // Device scope
            #expect(constants[words[scope + 1]] == 0) // Relaxed ordering
            if opcode == 230 {
                try #require(scope + 2 < offset + count)
                #expect(constants[words[scope + 2]] == 0) // Relaxed compare-exchange failure ordering
            }
        }
        offset += count
    }
    let extrema: Set<UInt32> = scalar == .i32 ? [236, 238] : [237, 239]
    #expect(opcodes == extrema.union([227, 228, 229, 230, 234, 235, 240, 241, 242]))
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], AtomicExportScalar.allCases)
private func spirvAtomicCodeGeneratorRoundTripsRelaxedStorageAtomics(_ format: ShaderOutputFormat, _ scalar: AtomicExportScalar) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportStorageAtomics(scalar, with: compiler)
    try expectRelaxedAtomicInstructions(words, scalar: scalar)
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [4, 56])
    #expect(shader.storageBindings.map(\.isReadOnly) == [false, false])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(text.contains("atomic<\(scalar.rawValue)>"))
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

#if canImport(Metal)
@Test(arguments: AtomicExportScalar.allCases)
private func spirvAtomicCodeGeneratorReturnsOldValuesAndFlagsOnMetal(_ scalar: AtomicExportScalar) async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportStorageAtomics(scalar, with: compiler)
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let counter = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let output = try await device.createBuffer(.init(size: 56, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 56, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): counter, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 56), .read(buffer: readback, byteCount: 56),
            .copy(source: counter, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
        ]).wait()
        let minResult: UInt32 = scalar == .i32 ? 0xfffffffe : 2
        let expected: [UInt32] = [0xfffffffc, 0xfffffffc, 3, 0xfffffffe, minResult, 5, 4, 7, 5, 11, 1, 17, 17, 0]
        #expect(snapshots == [expected, [17]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func spirvAtomicCodeGeneratorGuardsDescriptorArrayAtomicsOnMetal() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct Record { head: u32, counter: atomic<u32> }
        @group(0) @binding(0) var<storage, read_write> records: binding_array<Record>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;

        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            let index = i32(lane) - 1i;
            let cas = atomicCompareExchangeWeak(&records[index].counter, 0u, 11u);
            let previous = atomicAdd(&records[index].counter, 3u);
            atomicXor(&records[index].counter, 1u);
            output[lane] = vec4u(cas.old_value, select(0u, 1u, cas.exchanged), previous,
                                 atomicLoad(&records[index].counter));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let layout: [IRResourceBinding: UInt32] = [.init(group: 0, binding: 0): 2]
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv, bindingArrayLengths: layout) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    #expect(shader.storageBindings.first?.bindingArray?.count == 2)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let records = try await device.createBuffer(.init(size: 512, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: records, offset: 0, words: [91, 0]),
            .write(buffer: records, offset: 256, words: [92, 7]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0, arrayElement: 0): .init(buffer: records, size: 8),
                .init(binding: 0, arrayElement: 1): .init(buffer: records, offset: 256, size: 8),
                .init(binding: 1): .init(buffer: output),
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
            .copy(source: records, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
            .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
        ]).wait()
        let expected: [UInt32] = [0, 0, 0, 0, 0, 1, 11, 15, 7, 0, 7, 11, 0, 0, 0, 0]
        #expect(snapshots == [expected, [91, 15], [92, 11]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
