import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func logicalSubgroupModule(
        opcode: UInt32 = 362, operation: UInt32 = 0, vector: Bool = true,
        arithmeticCapability: Bool = true, numericResult: Bool = false
    ) -> [UInt32] {
        var words = head
        let type: UInt32 = numericResult ? 2 : vector ? 4 : 3
        for instruction in [
            i(17, 1), i(17, 61), arithmeticCapability ? i(17, 63) : [], i(14, 0, 1),
            i(15, 5, 20, 0x6e69616d, 0, 18), i(16, 20, 17, 4, 1, 1),
            i(71, 18, 11, 41), i(71, 5, 6, 4), i(71, 6, 2), i(72, 6, 0, 35, 0),
            i(71, 17, 34, 0), i(71, 17, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(20, 3), i(23, 4, 3, 2),
            i(43, 2, 30, 0), i(43, 2, 31, 1), i(43, 2, 32, 2), i(43, 2, 33, 3), i(43, 2, 34, 8),
            i(28, 5, 2, 34), i(30, 6, 5), i(32, 7, 12, 6), i(32, 8, 12, 2),
            i(32, 9, 1, 2), i(33, 10, 1), i(59, 7, 17, 12), i(59, 9, 18, 1),
            i(54, 1, 20, 0, 10), i(248, 21), i(61, 2, 40, 18), i(176, 3, 41, 40, 33), i(170, 3, 42, 40, 31),
            vector ? i(80, 4, 43, 41, 42) : [],
            i(opcode, type, 44, 33, operation, vector ? 43 : 41),
            vector ? i(81, 3, 45, 44, 0) : i(83, 3, 45, 44),
            vector ? i(81, 3, 46, 44, 1) : i(83, 3, 46, 42),
            i(169, 2, 47, 45, 31, 30), i(169, 2, 48, 46, 31, 30),
            i(132, 2, 49, 40, 32), i(128, 2, 50, 49, 31),
            i(65, 8, 51, 17, 30, 49), i(65, 8, 52, 17, 30, 50), i(62, 51, 47), i(62, 52, 48),
            i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvLogicalSubgroupsRejectInvalidCapabilityOperationAndType() throws {
        try fails(Self.logicalSubgroupModule(arithmeticCapability: false), code: .unsupportedCapability)
        try fails(Self.logicalSubgroupModule(operation: 3), code: .unsupportedConstruct)
        try fails(Self.logicalSubgroupModule(numericResult: true), code: .typeMismatch)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func spirvLogicalSubgroupsTranslateReductionsAndScans(_ format: ShaderOutputFormat, _ vector: Bool) throws {
    let compiler = IndependentShaderCompiler()
    for opcode: UInt32 in 362...364 {
        for operation: UInt32 in 0...2 {
            let shader = try compiler.compileCompute(
                .spirv(SPIRVFrontendTests.logicalSubgroupModule(opcode: opcode, operation: operation, vector: vector)), to: format)
            #expect(shader.requiresSubgroups)
            #expect(shader.storageBindings.map(\.minimumByteSize) == [32])
            guard case .text(let actual, let text) = shader.artifact else { throw ShaderFailure.invalidHandle }
            #expect(actual == format)
            if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.subgroup]) }
        }
    }
}

#if canImport(Metal)
@Test func spirvLogicalSubgroupsPreservePrefixesAndIdentitiesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.subgroups])
    do {
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = []
        for opcode: UInt32 in 362...364 {
            for operation: UInt32 in 0...2 {
                let words = SPIRVFrontendTests.logicalSubgroupModule(opcode: opcode, operation: operation)
                let pipeline = try await device.createComputePipeline(compiler.compileCompute(.spirv(words), to: .msl))
                commands.append(.dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)))
                commands.append(.copy(source: output, destination: readback, byteCount: 32))
                commands.append(.read(buffer: readback, byteCount: 32))
            }
        }
        let snapshots = try await device.submitCommands(commands).wait()
        #expect(snapshots == [
            [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 1, 1, 1],
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
