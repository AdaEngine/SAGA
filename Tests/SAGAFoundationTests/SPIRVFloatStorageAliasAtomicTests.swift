import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func floatStorageAliasAtomicModule(
        floatFirst: Bool, floatAdd: Bool = true, capability: Bool = true, extensionDeclared: Bool = true
    ) -> [UInt32] {
        var words = head + i(17, 1)
        if floatAdd && capability { words += i(17, 6033) }
        if floatAdd && extensionDeclared {
            // SPV_EXT_shader_atomic_float_add with a NUL terminator.
            words += i(10, 0x5f565053, 0x5f545845, 0x64616873, 0x615f7265, 0x696d6f74, 0x6c665f63, 0x5f74616f, 0x646461)
        }
        words += i(14, 0, 1) + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        for block: UInt32 in 4...6 {
            words += i(71, block, 2)
            for member: UInt32 in 0..<(block == 6 ? 7 : 3) { words += i(72, block, member, 35, member * 4) }
        }
        for variable: UInt32 in [15, 16] { words += i(71, variable, 34, 0) + i(71, variable, 33, 0) + i(71, variable, 20) }
        words += i(71, 17, 34, 0) + i(71, 17, 33, 1)
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(30, 4, 3, 3, 3), i(30, 5, 2, 2, 2),
            i(30, 6, 2, 2, 2, 2, 2, 2, 2), i(32, 7, 12, 4), i(32, 8, 12, 5), i(32, 9, 12, 6),
            i(32, 10, 12, 3), i(32, 11, 12, 2), i(33, 12, 1),
        ] { words += instruction }
        for value: UInt32 in 0..<7 { words += i(43, 2, 20 + value, value) }
        words += i(43, 3, 27, 0x3f800000) + i(43, 3, 28, 0x3f000000) + i(43, 3, 29, 0x40e00000) + i(43, 3, 30, 0x80000000)
        words += i(43, 2, 31, floatAdd ? 0x3fc00003 : 0x3f800003) + i(43, 2, 32, 0x80000000)
        let floatVariable = i(59, 7, 15, 12)
        let integerVariable = i(59, 8, 16, 12)
        words += floatFirst ? floatVariable + integerVariable : integerVariable + floatVariable
        for instruction in [
            i(59, 9, 17, 12), i(54, 1, 40, 0, 12), i(248, 41),
            i(65, 10, 42, 15, 20), i(65, 11, 43, 16, 20), i(65, 10, 44, 15, 21), i(65, 11, 45, 16, 21),
            i(228, 42, 21, 20, 27),
            floatAdd ? i(6035, 3, 50, 42, 21, 20, 28) : i(227, 3, 50, 42, 21, 20),
            i(234, 2, 51, 43, 21, 20, 21), i(241, 2, 52, 43, 21, 20, 22), i(230, 2, 53, 43, 21, 20, 20, 32, 31),
            i(229, 3, 54, 42, 21, 20, 29), i(227, 3, 55, 42, 21, 20), i(62, 44, 30), i(61, 2, 56, 45),
        ] { words += instruction }
        for offset: UInt32 in 0..<7 {
            let raw = 50 + offset
            let floatResult = offset == 0 || offset == 4 || offset == 5
            let bits = floatResult ? 60 + offset : raw
            if floatResult { words += i(124, 2, bits, raw) }
            words += i(65, 11, 70 + offset, 17, 20 + offset) + i(62, 70 + offset, bits)
        }
        return words + i(253) + i(56)
    }

    @Test(arguments: [true, false])
    func spirvFloatStorageAliasAtomicAddRequiresExtensionAndCapability(_ floatFirst: Bool) throws {
        try fails(Self.floatStorageAliasAtomicModule(floatFirst: floatFirst, capability: false), code: .unsupportedCapability)
        try fails(Self.floatStorageAliasAtomicModule(floatFirst: floatFirst, extensionDeclared: false), code: .unsupportedCapability)
    }
}

@Test(arguments: [true, false], [true, false])
func spirvFloatStorageAliasAtomicOperationsTranslate(_ floatFirst: Bool, _ floatAdd: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.floatStorageAliasAtomicModule(floatFirst: floatFirst, floatAdd: floatAdd))
    if !floatAdd { _ = try compiler.parseAndValidate(input) }
    for format: ShaderOutputFormat in [.msl, .hlsl, .glsl, .wgsl] {
        let shader = try compiler.compileCompute(input, to: format)
        #expect(shader.storageBindings.map(\.binding) == [IRResourceBinding(group: 0, binding: 0), .init(group: 0, binding: 1)])
        #expect(shader.storageBindings.map(\.minimumByteSize) == [12, 28])
        guard case .text(let actual, let code) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(actual == format)
        if format == .wgsl {
            #expect(code.contains("atomic<u32>"))
            _ = try compiler.parseAndValidate(.wgsl(code))
            _ = try compiler.compileWebGPUCompute(input)
        }
    }
}

#if canImport(Metal)
@Test(arguments: [true, false], [true, false])
func spirvFloatStorageAliasAtomicOperationsPreservePreviousBitsOnMetal(_ floatFirst: Bool, _ floatAdd: Bool) async throws {
    let words = SPIRVFrontendTests.floatStorageAliasAtomicModule(floatFirst: floatFirst, floatAdd: floatAdd)
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
    let integerStart: UInt32 = floatAdd ? 0x3fc00000 : 0x3f800000
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let storage = try await device.createBuffer(.init(size: 12, usage: [.storage, .copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 28, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 28, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: storage, offset: 0, words: [0, 0, 0xdeadbeef]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): storage, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 28), .read(buffer: readback, byteCount: 28),
            .copy(source: storage, destination: readback, byteCount: 12), .read(buffer: readback, byteCount: 12),
        ]).wait()
        #expect(snapshots == [
            [0x3f800000, integerStart, integerStart + 1, integerStart + 3, 0x80000000, 0x40e00000, 0x80000000],
            [0x40e00000, 0x80000000, 0xdeadbeef],
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
