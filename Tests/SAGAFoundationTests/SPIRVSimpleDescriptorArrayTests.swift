import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func simpleDescriptorArrayModule(
        runtime: Bool, memoryModel: UInt32 = 0, stride: UInt32? = 4,
        dynamicIndex: Bool = false, dynamicCapability: Bool = false
    ) -> [UInt32] {
        var words = head
        words[1] = 0x00010600
        words += i(17, 1)
        if dynamicCapability { words += i(17, 29) } // SampledImageArrayDynamicIndexing
        words += i(14, 0, memoryModel) + i(15, 4, 30, 0x6e69616d, 0, 22, 23, 24) + i(16, 30, 7)
        if let stride { words += i(71, 18, 6, stride) + i(71, 19, 6, stride) }
        words += i(71, 22, 34, 0) + i(71, 22, 33, 0) + i(71, 23, 34, 0) + i(71, 23, 33, 1) + i(71, 24, 30, 0)
        for instruction in [
            i(19, 1), i(21, 2, 32, 1), i(22, 3, 32), i(23, 4, 3, 2), i(23, 5, 3, 4),
            i(25, 6, 3, 1, 2, 0, 0, 1, 0), i(26, 7), i(27, 8, 6), i(32, 9, 0, 6), i(32, 10, 0, 7), i(32, 11, 3, 5),
            i(43, 2, 12, 1), i(43, 2, 13, 0), i(43, 3, 14, 0), i(43, 3, 15, 0x3f000000),
            i(44, 4, 16, 15, 15), i(43, 2, 17, 2),
        ] { words += instruction }
        words += runtime ? i(29, 18, 6) + i(29, 19, 7) : i(28, 18, 6, 17) + i(28, 19, 7, 17)
        for instruction in [
            i(32, 20, 0, 18), i(32, 21, 0, 19), i(33, 25, 1),
            i(59, 20, 22, 0), i(59, 21, 23, 0), i(59, 11, 24, 3), i(54, 1, 30, 0, 25), i(248, 31),
        ] { words += instruction }
        if dynamicIndex { words += i(128, 2, 32, 12, 13) }
        let index: UInt32 = dynamicIndex ? 32 : 12
        for instruction in [
            i(65, 9, 33, 22, index), i(66, 10, 34, 23, index), i(61, 6, 35, 33), i(61, 7, 36, 34),
            i(86, 8, 37, 35, 36), i(88, 5, 38, 37, 16, 2, 14), i(62, 24, 38), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test(arguments: [false, true])
    func spirvSimpleModelNormalizesOpaqueDescriptorArrays(runtime: Bool) throws {
        let input = ShaderInput.spirv(Self.simpleDescriptorArrayModule(runtime: runtime))
        let compiler = IndependentShaderCompiler()
        let capabilities: ShaderCapabilities = [.textureAndSamplerBindingArray]
        let module = try compiler.parseAndValidate(input, capabilities: capabilities).module
        let resources = module.globalVariables.values.filter { $0.binding != nil }
        #expect(resources.count == 2)
        for global in resources {
            guard case .bindingArray(_, let size) = try module.types.value(for: global.type).kind else {
                Issue.record("Expected an opaque descriptor array")
                continue
            }
            #expect(size == (runtime ? .dynamic : .constant(2)))
        }

        guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
            throw ShaderFailure.incompatibleArtifact
        }
        _ = try compiler.parseAndValidate(.spirv(words), capabilities: capabilities)
    }

    @Test(arguments: [false, true], [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
    func spirvSimpleModelDescriptorSamplesReachTextBackends(runtime: Bool, format: ShaderOutputFormat) throws {
        let input = ShaderInput.spirv(Self.simpleDescriptorArrayModule(runtime: runtime))
        let layout: [IRResourceBinding: UInt32] = runtime ? [.init(group: 0, binding: 0): 2, .init(group: 0, binding: 1): 2] : [:]
        let compiler = IndependentShaderCompiler()
        let stage = try compiler.compileRenderStage(input, entryPoint: "main", to: format, bindingArrayLengths: layout)
        #expect(stage.sampledTextures.first?.bindingArray?.count == 2)
        #expect(stage.samplerBindings.first?.bindingArray?.count == 2)
    }

    @Test func spirvSimpleModelDescriptorCompatibilityDoesNotWeakenVulkanPaths() throws {
        try fails(Self.simpleDescriptorArrayModule(runtime: false, memoryModel: 1))
        try fails(Self.simpleDescriptorArrayModule(runtime: true, memoryModel: 1, stride: nil),
                  code: .unsupportedCapability)
        try fails(Self.simpleDescriptorArrayModule(runtime: false, stride: 0))
        try fails(Self.simpleDescriptorArrayModule(runtime: true, dynamicIndex: true), code: .unsupportedCapability)
        _ = try parse(Self.simpleDescriptorArrayModule(runtime: true, dynamicIndex: true, dynamicCapability: true))
    }
}
