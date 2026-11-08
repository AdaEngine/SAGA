import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func dynamicSubgroupBroadcastModule(version: UInt32 = 0x00010500, listStorage: Bool = true,
                                               dynamicSubgroup: Bool = true, dynamicQuad: Bool = true) -> [UInt32] {
        var words = head
        words[1] = version
        for instruction in [
            i(17, 1), i(17, 61), i(17, 64), i(17, 68), i(14, 0, 1),
            listStorage ? i(15, 5, 20, 0x6e69616d, 0, 18, 19, 17) : i(15, 5, 20, 0x6e69616d, 0, 18, 19),
            i(16, 20, 17, 4, 1, 1),
            i(71, 18, 11, 41), i(71, 19, 11, 26),
            i(71, 5, 6, 16), i(71, 6, 2), i(72, 6, 0, 35, 0), i(71, 17, 34, 0), i(71, 17, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(23, 3, 2, 3), i(23, 4, 2, 4), i(23, 12, 2, 2),
            i(43, 2, 30, 0), i(43, 2, 31, 1), i(43, 2, 33, 3), i(43, 2, 34, 4),
            i(43, 2, 35, 100), i(43, 2, 36, 1000), i(43, 2, 37, 8),
            i(28, 5, 4, 37), i(30, 6, 5), i(32, 7, 12, 6), i(32, 8, 12, 4),
            i(32, 9, 1, 2), i(32, 11, 1, 3), i(33, 10, 1),
            i(59, 7, 17, 12), i(59, 9, 18, 1), i(59, 11, 19, 1),
            i(54, 1, 20, 0, 10), i(248, 21),
            i(61, 2, 40, 18), i(128, 2, 41, 40, 35), i(128, 2, 42, 40, 36), i(80, 12, 43, 41, 42),
            i(61, 3, 44, 19), i(81, 2, 45, 44, 0), i(128, 2, 46, 45, 31),
            i(337, 2, 47, 33, 41, dynamicSubgroup ? 46 : 31), i(365, 12, 48, 33, 43, dynamicQuad ? 46 : 31),
            i(81, 2, 49, 48, 0), i(81, 2, 50, 48, 1),
            i(132, 2, 51, 45, 34), i(128, 2, 52, 51, 40),
            i(80, 4, 53, 47, 49, 50, 40), i(65, 8, 54, 17, 30, 52), i(62, 54, 53),
            i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvDynamicBroadcastRequiresVersion15AndListedGlobals() throws {
        for dynamicQuad in [false, true] {
            try fails(Self.dynamicSubgroupBroadcastModule(version: 0x00010400,
                                                       dynamicSubgroup: !dynamicQuad, dynamicQuad: dynamicQuad),
                      code: .unsupportedConstruct)
        }
        try fails(Self.dynamicSubgroupBroadcastModule(listStorage: false))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvDynamicBroadcastTranslatesBothModes(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = SPIRVFrontendTests.dynamicSubgroupBroadcastModule()
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.requiresSubgroups)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [128])
    guard case .text(let actual, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code), capabilities: [.subgroup]) }
}

#if canImport(Metal)
@Test func spirvDynamicBroadcastReadsRuntimeSelectedLanesOnMetal() async throws {
    let words = SPIRVFrontendTests.dynamicSubgroupBroadcastModule()
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 128, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 2)),
            .copy(source: output, destination: readback, byteCount: 128), .read(buffer: readback, byteCount: 128),
        ]).wait()
        var expected: [UInt32] = []
        for group: UInt32 in 0..<2 {
            for lane: UInt32 in 0..<4 {
                expected += [101 + group, 101 + group, 1001 + group, lane]
            }
        }
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
