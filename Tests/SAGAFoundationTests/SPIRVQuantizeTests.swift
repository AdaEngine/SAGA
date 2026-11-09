import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func quantizeModule(firstResult: UInt32 = 5) -> [UInt32] {
        var words = pre + i(15, 5, 20, 0x6e69616d, 0) + i(16, 20, 17, 1, 1, 1)
        for instruction in [
            i(71, 6, 2), i(72, 6, 0, 35, 0), i(72, 6, 1, 35, 16), i(72, 6, 2, 35, 32), i(72, 6, 3, 35, 48),
            i(71, 12, 34, 0), i(71, 12, 33, 0), i(71, 13, 34, 0), i(71, 13, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 2, 4), i(23, 5, 3, 4),
            i(30, 6, 4, 4, 4, 4), i(32, 7, 12, 6), i(32, 8, 12, 4), i(32, 9, 12, 2), i(33, 10, 1),
            i(43, 2, 14, 0), i(43, 2, 15, 1), i(43, 2, 16, 2), i(43, 2, 17, 3),
            i(59, 7, 12, 12), i(59, 7, 13, 12), i(54, 1, 20, 0, 10), i(248, 21),
        ] { words += instruction }
        for field: UInt32 in 0..<4 {
            let bits: UInt32 = field == 3 ? 2 : 4
            let value: UInt32 = field == 3 ? 3 : 5
            let pointer: UInt32 = field == 3 ? 9 : 8
            let id = 30 + field * 7
            for instruction in [
                field == 3 ? i(65, pointer, id, 12, 17, 14) : i(65, pointer, id, 12, 14 + field),
                i(61, bits, id + 1, id), i(124, value, id + 2, id + 1),
                i(116, field == 0 ? firstResult : value, id + 3, id + 2), i(124, bits, id + 4, id + 3),
                field == 3 ? i(65, pointer, id + 5, 13, 17, 14) : i(65, pointer, id + 5, 13, 14 + field),
                i(62, id + 5, id + 4),
            ] { words += instruction }
        }
        return words + i(253) + i(56)
    }

    @Test func spirvQuantizeRejectsWrongScalarKindAndShape() throws {
        try fails(Self.quantizeModule(firstResult: 4))
        try fails(Self.quantizeModule(firstResult: 3))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvQuantizeCompilesWithoutHalfCapability(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.quantizeModule()), to: format)
    #expect(!shader.requiresShaderF16)
    guard case .text(let target, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [])
        _ = try compiler.compileWebGPUCompute(.spirv(SPIRVFrontendTests.quantizeModule()))
    }
}

#if canImport(Metal)
@Test func spirvQuantizeKeepsInfinityAndFlushesHalfSubnormalsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.quantizeModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 52, usage: [.copyDestination, .mapRead]))
        let words: [UInt32] = [
            0x3f801000, 0x3f803000, 0x3ffff000, 0x477fe000,
            0x38800000, 0x387fffff, 0xb3800000, Float(65505).bitPattern,
            Float(-65505).bitPattern, 0x7f800000, 0xff800000, 0x7fc01234,
            0x80000000,
        ]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: words),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 52), .read(buffer: read, byteCount: 52),
        ]).wait()
        #expect(snapshots == [[
            0x3f800000, 0x3f804000, 0x40000000, 0x477fe000,
            0x38800000, 0, 0x80000000, 0x7f800000,
            0xff800000, 0x7f800000, 0xff800000, 0x7fc01234,
            0x80000000,
        ]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
