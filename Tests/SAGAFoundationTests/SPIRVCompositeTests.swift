import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func compositeModule(shuffleLane: UInt32 = 5, nestedLane: UInt32 = 3) -> [UInt32] {
        var words = pre + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        for instruction in [
            i(71, 10, 2), i(72, 10, 0, 35, 0), i(72, 10, 1, 35, 16), i(72, 10, 2, 35, 24),
            i(71, 11, 2), i(72, 11, 0, 35, 0), i(72, 11, 1, 35, 16), i(72, 11, 2, 35, 32),
            i(72, 11, 3, 35, 48), i(72, 11, 4, 35, 64),
            i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(21, 8, 32, 1), i(23, 3, 2, 2), i(23, 4, 2, 4), i(23, 9, 8, 2),
            i(43, 2, 30, 0), i(43, 2, 31, 1), i(43, 2, 32, 2), i(43, 2, 33, 3), i(43, 2, 34, 4),
            i(43, 2, 35, 900), i(43, 2, 36, 9), i(28, 5, 4, 32),
            i(30, 10, 4, 3, 9), i(30, 11, 4, 4, 4, 4, 4),
            i(32, 12, 12, 10), i(32, 13, 12, 11), i(32, 14, 12, 4), i(32, 15, 12, 3), i(32, 16, 12, 9),
            i(33, 17, 1), i(1, 3, 37), i(59, 12, 20, 12), i(59, 13, 21, 12), i(54, 1, 40, 0, 17), i(248, 41),
            i(65, 14, 42, 20, 30), i(61, 4, 43, 42), i(65, 15, 44, 20, 31), i(61, 3, 45, 44),
            i(65, 16, 46, 20, 32), i(61, 9, 47, 46), i(81, 8, 48, 47, 0),
            i(77, 2, 49, 43, 48), i(78, 4, 50, 43, 35, 48),
            i(79, 4, 51, 50, 45, shuffleLane, 0, 4, .max), i(80, 4, 52, 45, 49, 35),
            i(80, 5, 53, 50, 51), i(82, 5, 54, 36, 53, 1, nestedLane), i(81, 4, 55, 54, 1), i(81, 2, 56, 54, 0, 3),
            i(1, 3, 57), i(79, 3, 58, 37, 57, 0, 2), i(81, 2, 59, 58, 0), i(81, 2, 60, 58, 1),
            i(80, 4, 61, 49, 56, 59, 60),
        ] { words += instruction }
        for (index, value) in [UInt32(50), 51, 52, 55, 61].enumerated() {
            let field = UInt32(index)
            words += i(65, 14, 70 + field, 21, 30 + field) + i(62, 70 + field, value)
        }
        return words + i(253) + i(56)
    }

    @Test func spirvCompositeRejectsBadLaneAndNestedIndex() throws {
        try fails(Self.compositeModule(shuffleLane: 6))
        try fails(Self.compositeModule(nestedLane: 4))
        var words = Self.compositeModule()
        let insert = try #require(words.firstIndex(of: 6 << 16 | 78))
        words[insert + 5] = 45
        try fails(words)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvCompositeTranslatesMixedVectorAndNestedInsert(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.compositeModule()), to: format)
    guard case .text(let target, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(.spirv(SPIRVFrontendTests.compositeModule()))
    }
}

#if canImport(Metal)
@Test func spirvCompositeKeepsOriginalAndHandlesOutOfRangeOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.compositeModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = []
        for index: UInt32 in [2, .max, 4] {
            commands += [
                .write(buffer: input, offset: 0, words: [11, 22, 33, 44, 101, 202, index, 0]),
                .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                                 workgroups: .init(x: 1)),
                .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
            ]
        }
        let snapshots = try await device.submitCommands(commands).wait()
        let unchanged: [UInt32] = [11, 22, 33, 44, 202, 11, 101, 0, 101, 202, 0, 900, 202, 11, 101, 9, 0, 44, 0, 0]
        #expect(snapshots == [[11, 22, 900, 44, 202, 11, 101, 0, 101, 202, 33, 900, 202, 11, 101, 9, 33, 44, 0, 0],
                              unchanged, unchanged])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
