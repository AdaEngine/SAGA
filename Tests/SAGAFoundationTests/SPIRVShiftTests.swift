import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func shiftModule() -> [UInt32] {
        var words = pre + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        for (block, count): (UInt32, UInt32) in [(6, 2), (7, 4)] {
            words += i(71, block, 2)
            for member in 0..<count { words += i(72, block, member, 35, member * 16) }
        }
        for instruction in [
            i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(23, 4, 2, 4), i(23, 5, 3, 4),
            i(30, 6, 4, 5), i(30, 7, 4, 4, 4, 4), i(32, 8, 12, 6), i(32, 9, 12, 7),
            i(32, 10, 12, 4), i(32, 11, 12, 5), i(33, 12, 1),
            i(43, 2, 15, 0), i(43, 2, 16, 1), i(43, 2, 17, 2), i(43, 2, 18, 3),
            i(59, 8, 20, 12), i(59, 9, 21, 12), i(54, 1, 40, 0, 12), i(248, 41),
            i(65, 10, 42, 20, 15), i(61, 4, 43, 42), i(124, 5, 44, 43),
            i(65, 11, 45, 20, 16), i(61, 5, 46, 45),
            i(194, 4, 47, 44, 46), i(195, 4, 48, 43, 46), i(196, 4, 49, 44, 46),
            i(81, 2, 50, 43, 0), i(81, 3, 51, 46, 0), i(124, 3, 52, 50),
            i(194, 3, 53, 52, 51), i(124, 2, 54, 53), i(195, 2, 55, 50, 51), i(196, 2, 56, 52, 51),
            i(80, 4, 57, 54, 55, 56, 15),
            i(65, 10, 60, 21, 15), i(62, 60, 47), i(65, 10, 61, 21, 16), i(62, 61, 48),
            i(65, 10, 62, 21, 17), i(62, 62, 49), i(65, 10, 63, 21, 18), i(62, 63, 57), i(253), i(56),
        ] { words += instruction }
        return words
    }

    static func wideShiftModule() -> [UInt32] {
        var words = head + i(17, 1) + i(17, 11) + i(14, 0, 1)
        words += i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        let inputOffsets: [UInt32] = [0, 16, 24, 32]
        let outputOffsets: [UInt32] = [0, 16, 32, 48, 56, 64, 72]
        for (block, offsets): (UInt32, [UInt32]) in [(10, inputOffsets), (11, outputOffsets)] {
            words += i(71, block, 2)
            for (member, offset) in offsets.enumerated() { words += i(72, block, UInt32(member), 35, offset) }
        }
        for instruction in [
            i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(21, 4, 64, 0), i(21, 5, 64, 1),
            i(23, 6, 2, 2), i(23, 7, 3, 2), i(23, 8, 4, 2), i(23, 9, 5, 2),
            i(30, 10, 8, 7, 7, 9), i(30, 11, 9, 8, 8, 7, 7, 7, 7),
            i(32, 12, 12, 10), i(32, 13, 12, 11), i(32, 14, 12, 7), i(32, 15, 12, 8), i(32, 16, 12, 9), i(33, 17, 1),
        ] { words += instruction }
        for index: UInt32 in 0..<6 { words += i(43, 2, 22 + index, index) }
        for instruction in [
            i(59, 12, 20, 12), i(59, 13, 21, 12), i(54, 1, 40, 0, 17), i(248, 41),
            i(65, 15, 42, 20, 22), i(61, 8, 43, 42), i(65, 14, 44, 20, 23), i(61, 7, 45, 44),
            i(65, 14, 46, 20, 24), i(61, 7, 47, 46), i(65, 16, 48, 20, 25), i(61, 9, 49, 48),
            i(194, 9, 50, 43, 45), i(195, 8, 51, 43, 45), i(196, 8, 52, 43, 45),
            i(194, 7, 53, 47, 49), i(195, 7, 54, 47, 49), i(196, 7, 55, 47, 49),
            i(65, 16, 60, 21, 22), i(62, 60, 50), i(65, 15, 61, 21, 23), i(62, 61, 51),
            i(65, 15, 62, 21, 24), i(62, 62, 52), i(65, 14, 63, 21, 25), i(62, 63, 53),
            i(65, 14, 64, 21, 26), i(62, 64, 54), i(65, 14, 65, 21, 27), i(62, 65, 55), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvShiftsRejectWrongBaseWidthAndCountShape() throws {
        let module = Self.shiftModule()
        let logical = try #require(module.firstIndex(of: 5 << 16 | 194))
        var words = module
        words[logical + 4] = 16
        try fails(words)
        words = module
        words[logical + 1] = 2
        try fails(words)

        words = Self.wideShiftModule()
        let wide = try #require(words.firstIndex(of: 5 << 16 | 194))
        words[wide + 1] = 7
        try fails(words)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvShiftsReachAllTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fixture = ShaderInput.spirv(SPIRVFrontendTests.shiftModule())
    let shader = try compiler.compileCompute(fixture, to: format)
    guard case .text(let target, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(fixture)
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvShiftsRespectSignednessAndIndependentCountWidthOnMetal(_ wide: Bool) async throws {
    let words = wide ? SPIRVFrontendTests.wideShiftModule() : SPIRVFrontendTests.shiftModule()
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let inputWords: [UInt32] = wide
            ? [4, 0x80000000, 1, 0x80000000, 1, 63, 0x80000004, 0x80000002, 1, 0, 1, 1]
            : [0x80000004, 0x80000001, 0xffffffff, 7, 1, 31, 32, 0xffffffff]
        let expected: [UInt32] = wide
            ? [2, 0x40000000, 1, 0, 2, 0xc0000000, .max, .max, 8, 0, 0, 0x80000000,
               0x40000002, 0, 0xc0000002, 0, 8, 0]
            : [0x40000002, 1, 0, 0, 0xc0000002, .max, 0, 0, 8, 0x80000000, 0, 0, 0x40000002, 0xc0000002, 8, 0]
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: UInt64(inputWords.count * 4), usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: wide ? 80 : 64, usage: [.storage, .copySource]))
        let readSize = UInt64(expected.count * 4)
        let read = try await device.createBuffer(.init(size: readSize, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: inputWords),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: readSize), .read(buffer: read, byteCount: readSize),
        ]).wait()
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
