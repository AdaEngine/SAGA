import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func floatAtomicExportInput(arithmetic: Bool) -> ShaderInput {
    let update = arithmetic ? """
        output[2] = bitcast<u32>(atomicAdd(&data.values[0], 0.5));
        output[3] = bitcast<u32>(atomicSub(&data.values[0], 0.25));
        """ : """
        output[2] = bitcast<u32>(atomicLoad(&data.values[0]));
        output[3] = output[2];
        """
    return .wgsl("""
        struct Data { values: array<atomic<f32>, 2>, sentinel: u32 }
        @group(0) @binding(0) var<storage, read_write> data: Data;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 10>;
        var<workgroup> sharedValue: atomic<f32>;

        @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
            let initial = workgroupUniformLoad(&sharedValue);
            if lane == 0u {
                atomicStore(&sharedValue, -0.0);
                atomicStore(&data.values[0], -0.0);
                atomicStore(&data.values[1], 1.5);
            }
            let snapshot = workgroupUniformLoad(&sharedValue);
            let sharedBits = bitcast<u32>(atomicLoad(&sharedValue));
            if lane == 0u {
                output[0] = bitcast<u32>(atomicLoad(&data.values[0]));
                output[1] = bitcast<u32>(atomicExchange(&data.values[0], 0.75));
                \(update)
                output[4] = bitcast<u32>(atomicLoad(&data.values[0]));
                output[5] = bitcast<u32>(atomicExchange(&data.values[1], -0.0));
                output[6] = bitcast<u32>(atomicLoad(&data.values[1]));
                output[7] = bitcast<u32>(snapshot);
                output[8] = sharedBits;
                output[9] = bitcast<u32>(initial);
            }
        }
        """)
}

private func exportFloatAtomics(_ compiler: IndependentShaderCompiler, arithmetic: Bool) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(floatAtomicExportInput(arithmetic: arithmetic), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvFloatAtomicCodeGeneratorRequiresArithmeticCapabilityOnlyForAddAndSub(_ format: ShaderOutputFormat, _ arithmetic: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportFloatAtomics(compiler, arithmetic: arithmetic)
    try #require(words.count > 5)
    var capabilities: Set<UInt32> = []
    var extensions: Set<String> = []
    var opcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        if opcode == 17 && count == 2 { capabilities.insert(words[offset + 1]) }
        if opcode == 10 {
            let bytes = words[(offset + 1)..<(offset + count)].flatMap { word in
                (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) }
            }
            extensions.insert(String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self))
        }
        offset += count
    }
    #expect(opcodes.isSuperset(of: [227, 228, 229])) // AtomicLoad, AtomicStore, AtomicExchange
    #expect(opcodes.contains(6035) == arithmetic) // AtomicFAddEXT
    #expect(capabilities.contains(6033) == arithmetic)
    #expect(extensions.contains("SPV_EXT_shader_atomic_float_add") == arithmetic)
    let input = ShaderInput.spirv(words)
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [12, 40])
    #expect(shader.workgroupStorageByteSize == 16)
    if format == .wgsl {
        let imported = try compiler.parseAndValidate(input, capabilities: [.shaderFloat32Atomic]).module
        #expect(imported.types.values.contains { $0.kind == .atomic(.f32) })
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(text.contains("atomic<u32>"))
        #expect(text.contains(", 0x80000000u)"))
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

#if canImport(Metal)
private func runFloatAtomicExportOnMetal(_ shader: ComputeShaderArtifact, initial: [UInt32], outputSize: UInt64) async throws -> [[UInt32]] {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let dataSize = UInt64(initial.count * MemoryLayout<UInt32>.size)
        let data = try await device.createBuffer(.init(size: dataSize, usage: [.storage, .copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: outputSize, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: max(dataSize, outputSize), usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: data, offset: 0, words: initial),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: outputSize), .read(buffer: readback, byteCount: outputSize),
            .copy(source: data, destination: readback, byteCount: dataSize), .read(buffer: readback, byteCount: dataSize),
        ]).wait()
        await device.close()
        return snapshots
    } catch {
        await device.close()
        throw error
    }
}

@Test(arguments: [false, true])
func spirvFloatAtomicCodeGeneratorPreservesOldBitsOnMetal(_ arithmetic: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportFloatAtomics(compiler, arithmetic: arithmetic)
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    let snapshots = try await runFloatAtomicExportOnMetal(shader, initial: [0, 0, 0xfeedface], outputSize: 40)
    let final = Float(arithmetic ? 1.0 : 0.75).bitPattern
    let negZero: UInt32 = 0x80000000
    let expected = [negZero, negZero, Float(0.75).bitPattern, Float(arithmetic ? 1.25 : 0.75).bitPattern,
                    final, Float(1.5).bitPattern, negZero, negZero, negZero, 0]
    #expect(snapshots == [expected, [final, negZero, 0xfeedface]])
}

@Test func spirvFloatAtomicCodeGeneratorReexportsMixedAliasBitsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.floatStorageAliasAtomicModule(floatFirst: true))
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    #expect(shader.storageBindings.map(\.binding.binding) == [0, 1])
    #expect(shader.storageBindings.map(\.minimumByteSize) == [12, 28])
    let snapshots = try await runFloatAtomicExportOnMetal(shader, initial: [0, 0, 0xdeadbeef], outputSize: 28)
    #expect(snapshots == [
        [0x3f800000, 0x3fc00000, 0x3fc00001, 0x3fc00003, 0x80000000, 0x40e00000, 0x80000000],
        [0x40e00000, 0x80000000, 0xdeadbeef],
    ])
}
#endif
