import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func vulkanAtomicHeader(
        extensionForm: Bool, vulkan: Bool = true, deviceCapability: Bool = false
    ) -> [UInt32] {
        var words = head
        words[1] = extensionForm ? 0x00010300 : 0x00010500
        words += i(17, 1)
        if vulkan {
            words += i(17, 5345) // VulkanMemoryModel
            if deviceCapability { words += i(17, 5346) } // VulkanMemoryModelDeviceScope
            if extensionForm {
                words += i(10, 0x5f565053, 0x5f52484b, 0x6b6c7576, 0x6d5f6e61, 0x726f6d65, 0x6f6d5f79, 0x006c6564)
            }
        }
        return words + i(14, 0, vulkan ? 3 : 1)
    }

    private static func vulkanBufferAtomicModule(
        extensionForm: Bool = false, vulkan: Bool = true, deviceCapability: Bool = false,
        scope: UInt32 = 5, workgroup: Bool = false, semantics: UInt32 = 0,
        equal: UInt32 = 0, unequal: UInt32 = 0
    ) -> [UInt32] {
        var words = vulkanAtomicHeader(extensionForm: extensionForm, vulkan: vulkan, deviceCapability: deviceCapability)
        words += extensionForm ? i(15, 5, 20, 0x6e69616d, 0) : i(15, 5, 20, 0x6e69616d, 0, 16)
        words += i(16, 20, 17, 1, 1, 1)
        if !workgroup {
            words += i(71, 3, 2) + i(72, 3, 0, 35, 0) + i(71, 16, 34, 0) + i(71, 16, 33, 0)
        }
        words += i(19, 1) + i(21, 2, 32, 0)
        if workgroup {
            words += i(32, 6, 4, 2)
        } else {
            words += i(30, 3, 2) + i(32, 6, 12, 3) + i(32, 7, 12, 2)
        }
        for instruction in [
            i(33, 4, 1), i(43, 2, 8, scope), i(43, 2, 9, 0), i(43, 2, 10, 1),
            i(43, 2, 11, equal), i(43, 2, 12, unequal), i(43, 2, 13, semantics),
            i(59, 6, 16, workgroup ? 4 : 12), i(54, 1, 20, 0, 4), i(248, 21),
        ] { words += instruction }
        let pointer: UInt32 = workgroup ? 16 : 22
        if !workgroup { words += i(65, 7, 22, 16, 9) }
        for instruction in [
            i(227, 2, 23, pointer, 8, 13), i(228, pointer, 8, 13, 10),
            i(234, 2, 24, pointer, 8, 13, 23), i(230, 2, 25, pointer, 8, 11, 12, 24, 10),
            i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test(arguments: [false, true], [UInt32(1), 2, 5])
    func spirvVulkanRelaxedStorageAtomicsReachPortableWGSL(extensionForm: Bool, scope: UInt32) throws {
        let words = Self.vulkanBufferAtomicModule(extensionForm: extensionForm, deviceCapability: scope == 1, scope: scope)
        let compiler = IndependentShaderCompiler()
        let shader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "main")
        #expect(shader.storageBindings.map(\.minimumByteSize) == [4])
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains("atomicLoad(") && source.contains("atomicStore(") &&
                source.contains("atomicAdd(") && source.contains("atomicCompareExchangeWeak("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    }

    @Test func spirvVulkanAtomicScopesRequireModelAndCapability() throws {
        _ = try parse(Self.vulkanBufferAtomicModule(scope: 2, workgroup: true))
        _ = try parse(Self.vulkanBufferAtomicModule(vulkan: false, scope: 1))
        try fails(Self.vulkanBufferAtomicModule(scope: 1), code: .unsupportedCapability)
        try fails(Self.vulkanBufferAtomicModule(vulkan: false), code: .unsupportedConstruct)
        try fails(Self.vulkanBufferAtomicModule(workgroup: true), code: .unsupportedConstruct)
    }

    @Test func spirvVulkanAtomicsDoNotDiscardOrderingOrMemoryFlags() throws {
        try fails(Self.vulkanBufferAtomicModule(semantics: 0x42), code: .unsupportedConstruct) // Acquire | UniformMemory
        try fails(Self.vulkanBufferAtomicModule(equal: 0x2044), code: .unsupportedConstruct) // MakeAvailable | Release | UniformMemory
        // The pinned Naga fixture uses Workgroup scope and WorkgroupMemory with no order.
        // Vulkan requires a non-relaxed order when a storage class semantics bit is set.
        try fails(Self.vulkanBufferAtomicModule(scope: 2, equal: 0x100, unequal: 0x100), code: .unsupportedConstruct)
        try fails(Self.vulkanBufferAtomicModule(scope: 2, unequal: 0x100), code: .unsupportedConstruct)
        try fails(Self.vulkanBufferAtomicModule(unequal: 0x42), code: .unsupportedConstruct)
    }

    @Test func spirvVulkanImageAtomicsRequireDeviceScopeCapability() throws {
        let body = Self.imageAtomicModule().dropFirst(Self.pre.count)
        try fails(Self.vulkanAtomicHeader(extensionForm: true) + body, code: .unsupportedCapability)
        let words = Self.vulkanAtomicHeader(extensionForm: true, deviceCapability: true) + body
        _ = try parse(words)
        var queue = Array(body)
        let scope = Self.i(43, 2, 14, 1)
        let offset = try #require(queue.indices.dropLast(scope.count - 1).first {
            queue[$0..<($0 + scope.count)].elementsEqual(scope)
        })
        queue[offset + scope.count - 1] = 5
        _ = try parse(Self.vulkanAtomicHeader(extensionForm: true) + queue)
    }
}
