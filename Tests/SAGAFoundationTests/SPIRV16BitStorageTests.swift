import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    // Vulkan 1.1 storage classes and Block decorations, with independent packed layouts.
    // The normal fixture deliberately has no Int16 capability and does all arithmetic in 32 bits.
    static func int16StorageModule(
        uniform: Bool = false, capability: UInt32? = 4433, version: UInt32 = 0x00010300,
        extensions: [String] = [], int16Arithmetic: Bool = false, narrowAddition: Bool = false
    ) -> [UInt32] {
        let sourceSpace: UInt32 = uniform ? 2 : 12
        var words = [0x07230203, version, 0, 100, 0] + i(17, 1)
        if let capability { words += i(17, capability) }
        if int16Arithmetic { words += i(17, 22) }
        for name in extensions {
            let bytes = Array(name.utf8) + [0]
            let count = (bytes.count + 3) / 4
            words.append(UInt32(count + 1) << 16 | 10)
            for word in 0..<count {
                var packed: UInt32 = 0
                for (byte, value) in bytes.dropFirst(word * 4).prefix(4).enumerated() {
                    packed |= UInt32(value) << (byte * 8)
                }
                words.append(packed)
            }
        }
        words += i(14, 0, 1) + i(15, 5, 43, 0x6e69616d, 0) + i(16, 43, 17, 1, 1, 1)
        for instruction in [
            i(71, 8, 2), i(72, 8, 0, 35, 0), i(72, 8, 1, 35, 2), i(72, 8, 2, 35, 4),
            i(71, 9, 2), i(72, 9, 0, 35, 0), i(72, 9, 1, 35, 2), i(72, 9, 2, 35, 4),
            i(72, 9, 3, 35, 8), i(72, 9, 4, 35, 12), i(72, 9, 5, 35, 16),
            i(71, 41, 34, 0), i(71, 41, 33, 0), i(71, 42, 34, 0), i(71, 42, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(21, 4, 16, 1), i(21, 5, 16, 0),
            i(23, 6, 5, 2), i(23, 7, 2, 2), i(30, 8, 4, 5, 6), i(30, 9, 4, 5, 6, 3, 2, 7),
            i(32, 10, sourceSpace, 8), i(32, 11, 12, 9), i(32, 12, sourceSpace, 4),
            i(32, 13, sourceSpace, 5), i(32, 14, sourceSpace, 6), i(32, 15, 12, 4),
            i(32, 16, 12, 5), i(32, 17, 12, 6), i(32, 18, 12, 3), i(32, 19, 12, 2),
            i(32, 20, 12, 7), i(33, 21, 1),
        ] { words += instruction }
        for member: UInt32 in 0..<6 { words += i(43, 2, 30 + member, member) }
        words += i(43, 3, 36, 5) + i(43, 2, 37, 8192) + i(44, 7, 40, 31, 34)
        words += i(59, 10, 41, sourceSpace) + i(59, 11, 42, 12) + i(54, 1, 43, 0, 21) + i(248, 44)
        for instruction in [
            i(65, 12, 45, 41, 30), i(61, 4, 46, 45), i(65, 13, 47, 41, 31), i(61, 5, 48, 47),
            i(65, 14, 49, 41, 32), i(61, 6, 50, 49),
        ] { words += instruction }
        if narrowAddition { words += i(128, 4, 51, 46, 46) }
        for instruction in [
            i(114, 3, 52, narrowAddition ? 51 : 46), i(113, 2, 53, 48), i(113, 7, 54, 50),
            i(128, 3, 55, 52, 36), i(128, 2, 56, 53, 37), i(128, 7, 57, 54, 40),
            i(114, 4, 58, 55), i(113, 5, 59, 56), i(113, 6, 60, 57), i(83, 5, 61, 59),
        ] { words += instruction }
        for (index, value) in [UInt32(58), 61, 60, 55, 56, 57].enumerated() {
            let member = UInt32(index)
            words += i(65, 15 + member, 70 + member, 42, 30 + member) + i(62, 70 + member, value)
        }
        return words + i(253) + i(56)
    }

    @Test func spirv16BitStorageRequiresMemoryCapabilitiesAndVersion() throws {
        try fails(Self.int16StorageModule(capability: nil), code: .unsupportedCapability)
        try fails(Self.int16StorageModule(uniform: true, capability: 4433), code: .unsupportedCapability)
        try fails(Self.int16StorageModule(version: 0x00010200), code: .unsupportedCapability)
        try fails(Self.int16StorageModule(uniform: true, capability: 4434, version: 0x00010200), code: .unsupportedCapability)
    }

    @Test func spirv16BitStorageAcceptsPre13WithBothExtensions() throws {
        let extensions = ["SPV_KHR_16bit_storage", "SPV_KHR_storage_buffer_storage_class"]
        _ = try parse(Self.int16StorageModule(version: 0x00010200, extensions: extensions))
        _ = try parse(Self.int16StorageModule(uniform: true, capability: 4434, version: 0x00010200, extensions: extensions))
    }

    @Test func spirv16BitStorageDoesNotAuthorizeNarrowArithmetic() throws {
        try fails(Self.int16StorageModule(narrowAddition: true), code: .unsupportedCapability)
        _ = try parse(Self.int16StorageModule(int16Arithmetic: true, narrowAddition: true))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl], [false, true])
func spirv16BitStoragePreservesPackedStorageAndUniformLayouts(_ format: ShaderOutputFormat, _ uniform: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.int16StorageModule(uniform: uniform, capability: uniform ? 4434 : 4433))
    let artifact = try compiler.compileCompute(input, to: format)
    #expect(artifact.requiresShaderInt16)
    #expect(!artifact.requiresShaderF16)
    #expect(artifact.storageBindings.map(\.minimumByteSize) == (uniform ? [24] : [8, 24]))
    #expect(artifact.uniformBindings.map(\.minimumByteSize) == (uniform ? [8] : []))
    guard case .text(let target, let source) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    if format == .wgsl {
        #expect(source.contains("enable wgpu_int16;"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderInt16])
        #expect(throws: ShaderFailure.self) { _ = try compiler.compileWebGPUCompute(input) }
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirv16BitStorageReadsAndWritesPackedHalfwordsOnMetal(_ uniform: Bool) async throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.int16StorageModule(uniform: uniform, capability: uniform ? 4434 : 4433))
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createBuffer(.init(size: 8, usage: [uniform ? .uniform : .storage, .copyDestination]))
        let destination = try await device.createBuffer(.init(size: 24, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 24, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: source, offset: 0, words: [0xf123fffe, 0xfffe1234]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): source, .init(binding: 1): destination],
                             workgroups: .init(x: 1)),
            .copy(source: destination, destination: read, byteCount: 24), .read(buffer: read, byteCount: 24),
        ]).wait()
        #expect(snapshots == [[0x11230003, 0x00021235, 3, 69923, 4661, 65538]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
