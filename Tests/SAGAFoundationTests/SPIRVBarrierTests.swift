import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func storageBarrierModule(
        fence: Bool = false, control: Bool = true, scope: UInt32 = 2, semantics: UInt32 = 0x48
    ) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 11, 0x6e69616d, 0, 16), i(16, 11, 17, 128, 1, 1),
            i(71, 16, 11, 28), i(71, 5, 6, 4), i(71, 6, 2), i(72, 6, 0, 35, 0),
            i(71, 14, 34, 0), i(71, 14, 33, 0), i(71, 15, 34, 0), i(71, 15, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(23, 3, 2, 3), i(43, 2, 4, 384),
            i(28, 5, 2, 4), i(30, 6, 5), i(32, 7, 12, 6), i(32, 8, 12, 2), i(32, 9, 1, 3),
            i(32, 10, 1, 2), i(33, 12, 1), i(43, 2, 20, 0), i(43, 2, 21, 10), i(43, 2, 22, 64),
            i(43, 2, 23, 2), i(43, 2, 24, scope), i(43, 2, 25, semantics), i(43, 2, 26, 0x108),
            i(59, 7, 14, 12), i(59, 7, 15, 12), i(59, 9, 16, 1),
            i(54, 1, 11, 0, 12), i(248, 30), i(65, 10, 31, 16, 20), i(61, 2, 32, 31),
            i(65, 8, 33, 14, 20, 32), i(128, 2, 34, 32, 21), i(62, 33, 34),
        ] { words += instruction }
        if fence { words += i(225, 24, 25) }
        if control { words += i(224, 23, 24, fence ? 26 : 25) }
        return words + i(198, 2, 35, 32, 22) + i(65, 8, 36, 14, 20, 35) + i(61, 2, 37, 36)
            + i(65, 8, 38, 15, 20, 32) + i(62, 38, 37) + i(253) + i(56)
    }

    @Test func spirvStorageBarrierRejectsUnrepresentableScopesAndStandaloneFences() throws {
        try fails(Self.storageBarrierModule(scope: 1), code: .unsupportedConstruct)
        try fails(Self.storageBarrierModule(fence: true, control: false), code: .unsupportedConstruct)
    }

    @Test func spirvStorageBarrierKeepsAllDeclaredMemoryKinds() throws {
        let module = try parse(Self.storageBarrierModule(semantics: 0x948))
        let flags = module.functions.values.flatMap { function in
            function.statements.values.compactMap { statement -> IRBarrier? in
                if case .controlBarrier(let value) = statement.kind { return value }
                return nil
            }
        }
        #expect(flags == [[.workgroup, .storage, .texture]])
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func spirvStorageBarrierTranslatesControlAndAdjacentFences(_ format: ShaderOutputFormat, _ fence: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.storageBarrierModule(fence: fence)), to: format)
    guard case .text(let actual, let code) = shader.artifact else { Issue.record("Expected text shader"); return }
    #expect(actual == format)
    switch format {
    case .wgsl:
        #expect(code.contains("storageBarrier();"))
        if fence { #expect(code.contains("workgroupBarrier();")) }
        _ = try compiler.parseAndValidate(.wgsl(code))
    case .msl:
        #expect(code.contains("mem_flags::mem_device"))
    case .hlsl:
        #expect(code.contains(fence ? "AllMemoryBarrierWithGroupSync()" : "DeviceMemoryBarrierWithGroupSync()"))
    case .glsl:
        #expect(code.contains("memoryBarrierBuffer();"))
        #expect(code.contains("barrier();"))
    default: Issue.record("Unexpected format")
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvStorageBarrierMakesPeerWritesVisibleOnMetal(_ fence: Bool) async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.storageBarrierModule(fence: fence)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let bytes: UInt64 = 384 * 4
        let scratch = try await device.createBuffer(.init(size: bytes, usage: [.storage]))
        let output = try await device.createBuffer(.init(size: bytes, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: bytes, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): scratch, .init(binding: 1): output],
                             workgroups: .init(x: 3)),
            .copy(source: output, destination: readback, byteCount: bytes), .read(buffer: readback, byteCount: bytes),
        ]).wait()
        let expected = (0..<384).map { (index: Int) -> UInt32 in UInt32((index ^ 64) + 10) }
        #expect(result == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
