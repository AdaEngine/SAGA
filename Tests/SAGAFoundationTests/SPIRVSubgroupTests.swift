import SAGACore
import SAGATypes
import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func subgroupModule(scope: UInt32 = 3, shuffleCapability: Bool = true,
                               operation: UInt32 = 0, builtin: UInt32 = 41,
                               relaxedPrecision: [UInt32] = [], extra: [[UInt32]] = []) -> [UInt32] {
        var words = head
        for instruction in [
            i(17, 1), i(17, 61), i(17, 62), i(17, 63), i(17, 64), i(17, 66), i(17, 68),
            shuffleCapability ? i(17, 65) : [],
            i(14, 0, 1), i(15, 5, 20, 0x6e69616d, 0, 18), i(16, 20, 17, 4, 1, 1),
        ] { words += instruction }
        for id in relaxedPrecision { words += i(71, id, 0) }
        for instruction in [
            i(71, 18, 11, builtin), i(71, 5, 6, 16), i(71, 6, 2), i(72, 6, 0, 35, 0),
            i(71, 17, 34, 0), i(71, 17, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(20, 3), i(23, 4, 2, 4),
            i(46, 2, 30), i(43, 2, 31, 1), i(43, 2, 32, 2), i(43, 2, 33, scope),
            i(43, 2, 34, 4), i(43, 2, 35, 100),
            i(28, 5, 4, 34), i(30, 6, 5), i(32, 7, 12, 6), i(32, 8, 12, 4),
            i(32, 9, 1, 2), i(33, 10, 1), i(59, 7, 17, 12), i(59, 9, 18, 1),
            i(54, 1, 20, 0, 10), i(248, 21), i(61, 2, 40, 18), i(128, 2, 41, 40, 35),
            i(346, 2, 42, 33, 41, 31), i(176, 3, 43, 40, 32), i(339, 4, 44, 33, 43),
            i(81, 2, 45, 44, 0), i(349, 2, 46, 33, operation, 41),
        ] { words += instruction }
        for instruction in extra { words += instruction }
        for instruction in [
            i(80, 4, 47, 42, 45, 46, 40), i(65, 8, 48, 17, 30, 40), i(62, 48, 47),
            i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvSubgroupRejectsInvalidScopeCapabilityAndOperation() throws {
        try fails(Self.subgroupModule(scope: 2), code: .unsupportedConstruct)
        try fails(Self.subgroupModule(shuffleCapability: false), code: .unsupportedCapability)
        try fails(Self.subgroupModule(operation: 3), code: .unsupportedConstruct)
        try fails(Self.subgroupModule(extra: [Self.i(366, 2, 50, 33, 41, 34)]))
        try fails(Self.subgroupModule(extra: [Self.i(334, 2, 50, 33, 41)]), code: .typeMismatch)
        try fails(Self.subgroupModule(extra: [Self.i(339, 2, 50, 33, 43)]), code: .typeMismatch)
        try fails(Self.subgroupModule(extra: [Self.i(350, 2, 50, 33, 0, 41)]), code: .typeMismatch)
    }

    @Test func spirvSubgroupComputeBuiltinsRetainTheirBindings() throws {
        let builtins: [(UInt32, IRBuiltIn)] = [(36, .subgroupSize), (38, .numSubgroups),
                                              (40, .subgroupId), (41, .subgroupInvocationId)]
        for (number, builtin) in builtins {
            let module = try parse(Self.subgroupModule(builtin: number))
            let function = try module.functions.value(for: module.entryPoints[0].function)
            #expect(function.arguments.first?.binding == .builtIn(builtin))
        }
    }

    @Test func spirvSubgroupPrecisionHintPreservesFullWidth() throws {
        let compiler = IndependentShaderCompiler()
        let standard = try compiler.compileCompute(.spirv(Self.subgroupModule()), to: .wgsl)
        let relaxed = try compiler.compileCompute(.spirv(Self.subgroupModule(relaxedPrecision: [18, 40])), to: .wgsl)
        #expect(standard.artifact == relaxed.artifact)
        try fails(Self.subgroupModule(relaxedPrecision: [43]))
        try fails(Self.subgroupModule(relaxedPrecision: [18, 18]))
    }

    @Test func spirvSubgroupRegistersGatherVoteAndScanResults() throws {
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(Self.subgroupModule(extra: [
            Self.i(337, 2, 50, 33, 41, 30), Self.i(338, 2, 51, 33, 41),
            Self.i(345, 2, 52, 33, 41, 40), Self.i(347, 2, 53, 33, 41, 31),
            Self.i(348, 2, 54, 33, 41, 31), Self.i(365, 2, 55, 33, 41, 30),
            Self.i(366, 2, 56, 33, 41, 32), Self.i(334, 3, 57, 33, 43), Self.i(335, 3, 58, 33, 43),
            Self.i(349, 2, 59, 33, 1, 51), Self.i(351, 2, 60, 33, 2, 59),
        ])), capabilities: [.subgroup]).module
        let statements = module.functions.values.flatMap { $0.statements.values.map(\.kind) }
        #expect(statements.count(where: { if case .subgroupGather = $0 { true } else { false } }) == 8)
        #expect(statements.count(where: { if case .subgroupCollectiveOperation = $0 { true } else { false } }) == 5)
        #expect(statements.contains { if case .subgroupGather(.quadSwap(.diagonal), _, _) = $0 { true } else { false } })
        #expect(statements.contains { if case .subgroupCollectiveOperation(.mul, .exclusiveScan, _, _) = $0 { true } else { false } })
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvSubgroupOperationsTranslate(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.subgroupModule()), to: format)
    #expect(shader.requiresSubgroups)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64])
    guard case .text(let actual, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code), capabilities: [.subgroup]) }
}

#if canImport(Metal)
@Test func spirvSubgroupOperationsExchangeAndReduceOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.subgroupModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        ]).wait()
        #expect(snapshots == [[101, 3, 406, 0, 100, 3, 406, 1, 103, 3, 406, 2, 102, 3, 406, 3]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
