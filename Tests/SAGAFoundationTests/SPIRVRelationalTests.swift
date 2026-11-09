import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func relationalModule(half: Bool = false) -> [UInt32] {
        var words: [UInt32] = [0x07230203, 0x00010300, 0, 250, 0]
        words += i(17, 1)
        if half { words += i(17, 9) }
        words += i(14, 0, 1) + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        for instruction in [
            i(71, 9, 2), i(72, 9, 0, 35, 0), i(71, 10, 2), i(72, 10, 0, 35, 0), i(71, 11, 6, 4),
            i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(22, 3, half ? 16 : 32), i(23, 4, 2, 4), i(23, 5, 3, 4),
            i(20, 6), i(23, 7, 6, 4), i(23, 8, 2, 2), i(43, 2, 22, 24), i(28, 11, 2, 22),
            i(30, 9, 4), i(30, 10, 11), i(32, 12, 12, 9), i(32, 13, 12, 10),
            i(32, 14, 12, 4), i(32, 15, 12, 2), i(33, 16, 1),
            i(41, 6, 23), i(42, 6, 24), i(44, 7, 25, 23, 23, 23, 23), i(44, 7, 26, 24, 24, 24, 24),
        ] { words += instruction }
        for word: UInt32 in 0..<24 { words += i(43, 2, 50 + word, word) }
        words += i(59, 12, 20, 12) + i(59, 13, 21, 12) + i(54, 1, 40, 0, 16) + i(248, 41)
        words += i(65, 14, 100, 20, 50) + i(61, 4, 101, 100)
        if half { words += i(79, 8, 102, 101, 101, 0, 1) }
        words += i(124, 5, 103, half ? 102 : 101) + i(156, 7, 104, 103) + i(157, 7, 105, 103)
        for lane: UInt32 in 0..<4 {
            words += i(81, 3, 110 + lane, 103, lane)
            words += i(156, 6, 120 + lane, 110 + lane) + i(157, 6, 130 + lane, 110 + lane)
            words += i(81, 6, 140 + lane, 104, lane) + i(81, 6, 144 + lane, 105, lane)
        }
        let masks: [UInt32] = [104, 105, 25, 26]
        for (index, mask) in masks.enumerated() {
            let destination = 150 + UInt32(index * 2)
            words += i(154, 6, destination, mask) + i(155, 6, destination + 1, mask)
        }
        let predicates = Array(140...147) + Array(120...123) + Array(130...133) + Array(150...157)
        for (index, predicate) in predicates.enumerated() {
            let word = UInt32(index)
            words += i(169, 2, 170 + word, UInt32(predicate), 51, 50)
            words += i(65, 15, 220 + word, 21, 50, 50 + word) + i(62, 220 + word, 170 + word)
        }
        return words + i(253) + i(56)
    }

    @Test func spirvRelationalRejectsInvalidOperandsResultsAndKernelInstructions() throws {
        func changed(_ opcode: UInt32, _ operand: Int, _ replacement: UInt32) throws -> [UInt32] {
            var words = Self.relationalModule()
            let offset = try #require(words.firstIndex(of: 4 << 16 | opcode))
            words[offset + operand] = replacement
            return words
        }
        try fails(changed(154, 1, 7))
        try fails(changed(155, 3, 23))
        try fails(changed(156, 1, 6))
        try fails(changed(157, 3, 101))
        for opcode: UInt32 in 158...160 {
            try fails(changed(156, 0, 4 << 16 | opcode), code: .unsupportedConstruct)
        }
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl], [false, true])
func spirvRelationalCompilesReductionsAndClassifications(_ format: ShaderOutputFormat, _ half: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let fixture = ShaderInput.spirv(SPIRVFrontendTests.relationalModule(half: half))
    let shader = try compiler.compileCompute(fixture, to: format)
    #expect(shader.requiresShaderF16 == half)
    guard case .text(let target, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    #expect(source.contains("any("))
    #expect(source.contains("all("))
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: half ? [.shaderFloat16] : [])
        _ = try compiler.compileWebGPUCompute(fixture)
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvRelationalClassifiesSpecialValuesAndReducesMasksOnMetal(_ half: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.relationalModule(half: half)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 96, usage: [.copyDestination, .mapRead]))
        let special: [UInt32] = half ? [0xfc007c00, 0x0001fe00, 0, 0] : [0x7f800000, 0xff800000, 0xffc01234, 1]
        let zeroAndNaNs: [UInt32] = half ? [0x80000000, 0x7c01fe00, 0, 0] : [0, 0x80000000, 0xffc01234, 0x7f800001]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: special),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
            .write(buffer: input, offset: 0, words: zeroAndNaNs),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
        ]).wait()
        #expect(snapshots == [
            [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
