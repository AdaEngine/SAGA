import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing

#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    private static func barrierMemoryModelHeader(_ model: UInt32, deviceScope: Bool = false) -> [UInt32] {
        var words = head + i(17, 1)
        if model == 3 {
            words += i(17, 5345) // VulkanMemoryModel
            if deviceScope { words += i(17, 5346) }
            words += i(10, 0x5f565053, 0x5f52484b, 0x6b6c7576, 0x6d5f6e61, 0x726f6d65, 0x6f6d5f79, 0x006c6564)
        }
        return words + i(14, 0, model)
    }

    private static func scopedBarrierModule(
        model: UInt32, execution: UInt32, scope: UInt32, semantics: UInt32,
        fenceScope: UInt32? = nil, deviceScope: Bool = false
    ) -> [UInt32] {
        var words = barrierMemoryModelHeader(model, deviceScope: deviceScope)
        for instruction in [
            i(15, 5, 10, 0x6e69616d, 0), i(16, 10, 17, 64, 1, 1),
            i(19, 1), i(21, 2, 32, 0), i(33, 3, 1),
            i(43, 2, 4, execution), i(43, 2, 5, scope), i(43, 2, 6, semantics),
            i(43, 2, 7, fenceScope ?? scope), i(54, 1, 10, 0, 3), i(248, 11),
        ] { words += instruction }
        if fenceScope != nil { words += i(225, 7, 6) }
        return words + i(224, 4, 5, 6) + i(253) + i(56)
    }

    static func vulkanWorkgroupBarrierOrderingModule(
        version: UInt32 = 0x00010300, includeWorkgroup: Bool = false, includeStorage: Bool = false
    ) -> [UInt32] {
        var words = barrierMemoryModelHeader(3)
        words[1] = version
        var interface: [UInt32] = [10]
        if includeWorkgroup { interface.append(11) }
        if includeStorage { interface.append(12) }
        words += [UInt32(5 + interface.count) << 16 | 15, 5, 13, 0x6e69616d, 0] + interface
        for instruction in [
            i(16, 13, 17, 64, 1, 1),
            i(71, 10, 11, 29), i(71, 5, 6, 4), i(71, 6, 2), i(72, 6, 0, 35, 0),
            i(71, 12, 34, 0), i(71, 12, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(20, 3), i(43, 2, 4, 128),
            i(28, 5, 2, 4), i(30, 6, 5), i(32, 7, 12, 2), i(32, 8, 12, 6), i(33, 9, 1),
            i(43, 2, 14, 0), i(43, 2, 15, 2), i(43, 2, 16, 0x108),
            i(43, 2, 17, 7), i(43, 2, 18, 9), i(43, 2, 19, 64),
            i(32, 30, 1, 2), i(32, 31, 4, 2),
            i(59, 30, 10, 1), i(59, 31, 11, 4), i(59, 8, 12, 12),
            i(54, 1, 13, 0, 9), i(248, 40),
            i(61, 2, 41, 10), i(170, 3, 42, 41, 14),
            i(247, 44, 0), i(250, 42, 43, 44),
            i(248, 43), i(228, 11, 15, 14, 17), i(249, 44),
            i(248, 44), i(224, 15, 15, 16),
            i(227, 2, 45, 11, 15, 14), i(224, 15, 15, 16),
            i(65, 7, 46, 12, 14, 41), i(62, 46, 45),
            i(247, 48, 0), i(250, 42, 47, 48),
            i(248, 47), i(228, 11, 15, 14, 18), i(249, 48),
            i(248, 48), i(224, 15, 15, 16),
            i(227, 2, 49, 11, 15, 14), i(128, 2, 50, 41, 19),
            i(65, 7, 51, 12, 14, 50), i(62, 51, 49),
            i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test(arguments: [UInt32(0x108), 0x848, 0x948])
    func spirvSimpleMemoryModelPreservesLegacyBarrierKinds(_ semantics: UInt32) throws {
        let scope: UInt32 = semantics == 0x108 ? 2 : 1
        let module = try parse(Self.scopedBarrierModule(
            model: 0, execution: 2, scope: scope, semantics: semantics, fenceScope: scope))
        let expected: IRBarrier = switch semantics {
        case 0x108: .workgroup
        case 0x848: [.storage, .texture]
        default: [.workgroup, .storage, .texture]
        }
        #expect(Self.barrierFlags(in: module) == [expected])
    }

    @Test func spirvSimpleMemoryModelPreservesSubgroupRendezvous() throws {
        let words = Self.scopedBarrierModule(model: 0, execution: 3, scope: 3, semantics: 0x88, fenceScope: 3)
        #expect(Self.barrierFlags(in: try parse(words)) == [.subgroup])
        let shader = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .wgsl)
        guard case .text(_, let source) = shader.artifact else { Issue.record("Expected WGSL"); return }
        #expect(source.contains("subgroupBarrier();"))
        try fails(Self.scopedBarrierModule(model: 0, execution: 3, scope: 3, semantics: 0x88, fenceScope: 2),
                  code: .unsupportedConstruct)
    }

    @Test func spirvSimpleMemoryModelRejectsObservableDeviceScopeNarrowing() throws {
        var words = Self.storageBarrierModule(fence: true, scope: 1)
        words[9] = 0 // Change GLSL450 to Simple, leaving its storage resources present.
        try fails(words, code: .unsupportedConstruct)
    }

    @Test func spirvVulkanMemoryModelPreservesWorkgroupBarriers() throws {
        let words = Self.vulkanWorkgroupBarrierOrderingModule()
        #expect(Self.barrierFlags(in: try parse(words)) == [.workgroup, .workgroup, .workgroup])
        let shader = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(words), entryPoint: "main")
        guard case .text(_, let source) = shader.artifact else { Issue.record("Expected WGSL"); return }
        #expect(source.contains("workgroupBarrier();"))
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
    }

    @Test func spirvEntryInterfaceRequiresUsedStorageAndWorkgroupFromVersion14() throws {
        let legacy = try parse(Self.vulkanWorkgroupBarrierOrderingModule())
        let modern = try parse(Self.vulkanWorkgroupBarrierOrderingModule(
            version: 0x00010400, includeWorkgroup: true, includeStorage: true))
        #expect(Self.barrierFlags(in: legacy) == Self.barrierFlags(in: modern))
        try fails(Self.vulkanWorkgroupBarrierOrderingModule(includeWorkgroup: true, includeStorage: true),
                  code: .unsupportedConstruct)

        for includeWorkgroup in [false, true] {
            let incomplete = Self.vulkanWorkgroupBarrierOrderingModule(
                version: 0x00010400, includeWorkgroup: includeWorkgroup, includeStorage: !includeWorkgroup)
            do {
                _ = try parse(incomplete)
                Issue.record("Accepted an incomplete SPIR-V 1.4 interface")
            } catch ShaderFailure.diagnostics(let diagnostics) {
                #expect(diagnostics.first?.code == .invalidBinary)
                #expect(diagnostics.first?.message == "Entry point omits a used interface variable")
            }
        }
    }

    @Test func spirvVulkanMemoryModelRejectsUnrepresentableBarrierSemantics() throws {
        for (execution, scope, semantics): (UInt32, UInt32, UInt32) in [
            (2, 2, 0), (2, 2, 0x104), (2, 2, 0x6108), (2, 5, 0x108), (3, 2, 0x108),
        ] {
            try fails(Self.scopedBarrierModule(model: 3, execution: execution, scope: scope, semantics: semantics),
                      code: .unsupportedConstruct)
        }
        try fails(Self.scopedBarrierModule(model: 3, execution: 2, scope: 1, semantics: 0x108),
                  code: .unsupportedCapability)
        try fails(Self.scopedBarrierModule(model: 3, execution: 2, scope: 1, semantics: 0x108, deviceScope: true),
                  code: .unsupportedConstruct)
    }

    private static func barrierFlags(in module: IRModule) -> [IRBarrier] {
        module.functions.values.flatMap { function in
            function.statements.values.compactMap { statement in
                if case .controlBarrier(let flags) = statement.kind { return flags }
                return nil
            }
        }
    }
}

#if canImport(Metal)
@Test func spirvVulkanMemoryModelOrdersPeerReadsBeforeOverwriteOnMetal() async throws {
    let binary = SPIRVFrontendTests.vulkanWorkgroupBarrierOrderingModule()
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(binary), entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let bytes: UInt64 = 128 * 4
        let output = try await device.createBuffer(.init(size: bytes, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: bytes, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: bytes), .read(buffer: readback, byteCount: bytes),
        ]).wait()
        #expect(result == [Array(repeating: 7, count: 64) + Array(repeating: 9, count: 64)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
