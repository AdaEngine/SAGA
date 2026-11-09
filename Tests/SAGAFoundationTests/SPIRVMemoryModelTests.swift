import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func memoryModelCompute(
        _ model: UInt32, version: UInt32 = 0x00010500, addressing: UInt32 = 0,
        capabilities: [UInt32] = [], legacyExtension: Bool = false, duplicateModel: Bool = false,
        decoration: UInt32? = nil
    ) -> [UInt32] {
        var words = head + i(17, 1)
        words[1] = version
        for capability in capabilities { words += i(17, capability) }
        if legacyExtension {
            // Null-terminated SPV_KHR_vulkan_memory_model.
            words += i(10, 0x5f565053, 0x5f52484b, 0x6b6c7576, 0x6d5f6e61, 0x726f6d65, 0x6f6d5f79, 0x006c6564)
        }
        words += i(14, addressing, model)
        if duplicateModel { words += i(14, addressing, model) }
        words += i(15, 5, 10, 0x6e69616d, 0) + i(16, 10, 17, 2, 3, 4)
        if let decoration {
            words += i(71, 5, 2) + i(72, 5, 0, 35, 0)
            words += i(71, 7, 33, 0) + i(71, 7, 34, 0) + i(71, 7, decoration)
        }
        words += i(19, 1) + i(33, 2, 1)
        if decoration != nil {
            words += i(21, 4, 32, 0) + i(30, 5, 4) + i(32, 6, 12, 5) + i(59, 6, 7, 12)
        }
        return words + i(54, 1, 10, 0, 2) + i(248, 11) + i(253) + i(56)
    }

    @Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
    func spirvMemoryModelsCompileWithoutOrderingSpecificOperations(format: ShaderOutputFormat) throws {
        let inputs = [
            Self.memoryModelCompute(0), // Simple
            Self.memoryModelCompute(1), // GLSL450
            Self.memoryModelCompute(3, capabilities: [5345]), // Core Vulkan
            Self.memoryModelCompute(3, version: 0x00010300, capabilities: [5345], legacyExtension: true),
            Self.memoryModelCompute(3, capabilities: [5345, 5346]),
        ]
        for words in inputs {
            let shader = try IndependentShaderCompiler().compileCompute(.spirv(words), to: format)
            #expect(shader.workgroupSize == .init(x: 2, y: 3, z: 4))
        }
    }

    @Test func spirvMemoryModelChecksCapabilitiesVersionAndAddressingModel() throws {
        for words in [
            Self.memoryModelCompute(3),
            Self.memoryModelCompute(1, capabilities: [5345]),
            Self.memoryModelCompute(3, capabilities: [5346]),
            Self.memoryModelCompute(1, capabilities: [5346]),
            Self.memoryModelCompute(3, version: 0x00010400, capabilities: [5345]),
        ] {
            try fails(words, code: .unsupportedCapability)
        }
        try fails(Self.memoryModelCompute(1, duplicateModel: true))
        try fails(Self.memoryModelCompute(1, addressing: 1), code: .unsupportedConstruct)
        try fails(Self.memoryModelCompute(2), code: .unsupportedConstruct) // OpenCL
    }

    @Test(arguments: [UInt32(21), 23])
    func spirvVulkanMemoryModelRejectsLegacyCoherentAndVolatileDecorations(decoration: UInt32) throws {
        _ = try parse(Self.memoryModelCompute(1, decoration: decoration))
        try fails(Self.memoryModelCompute(3, capabilities: [5345], decoration: decoration))
    }
}
