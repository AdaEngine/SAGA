import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func floatComparisonModule(vector: Bool = true, half: Bool = false,
                                      wrongResult: Bool = false, wrongOperand: Bool = false) -> [UInt32] {
        let laneCount: UInt32 = vector ? 4 : 1
        let wordCount = laneCount * 12
        let operandType: UInt32 = vector ? 5 : 3
        let predicateType: UInt32 = vector ? 7 : 6
        let inputType: UInt32 = half ? (vector ? 8 : 2) : (vector ? 4 : 2)
        let inputSize: UInt32 = half && vector ? 8 : laneCount * 4
        var words: [UInt32] = [0x07230203, 0x00010300, 0, 500, 0]
        words += i(17, 1)
        if half { words += i(17, 9) }
        words += i(14, 0, 1) + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        for instruction in [
            i(71, 10, 2), i(72, 10, 0, 35, 0), i(72, 10, 1, 35, inputSize),
            i(71, 11, 2), i(72, 11, 0, 35, 0), i(71, 12, 6, 4),
            i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(22, 3, half ? 16 : 32), i(23, 4, 2, 4), i(23, 5, 3, 4),
            i(20, 6), i(23, 7, 6, 4), i(23, 8, 2, 2),
            i(43, 2, 22, wordCount), i(28, 12, 2, 22), i(30, 10, inputType, inputType), i(30, 11, 12),
            i(32, 13, 12, 10), i(32, 14, 12, 11), i(32, 15, 12, inputType), i(32, 16, 12, 2), i(33, 17, 1),
        ] { words += instruction }
        if half && !vector { words += i(23, 9, 3, 2) }
        for word in 0..<wordCount { words += i(43, 2, 50 + word, word) }
        words += i(59, 13, 20, 12) + i(59, 14, 21, 12) + i(54, 1, 40, 0, 17) + i(248, 41)
        words += i(65, 15, 100, 20, 50) + i(61, inputType, 101, 100)
        words += i(65, 15, 102, 20, 51) + i(61, inputType, 103, 102)
        if half && !vector {
            words += i(124, 9, 106, 101) + i(81, 3, 104, 106, 0)
            words += i(124, 9, 107, 103) + i(81, 3, 105, 107, 0)
        } else {
            words += i(124, operandType, 104, 101) + i(124, operandType, 105, 103)
        }
        for offset: UInt32 in 0..<12 {
            let opcode = 180 + offset
            let comparison = 110 + offset
            let result = wrongResult && offset == 0 ? (vector ? 6 : 7) : predicateType
            words += i(opcode, result, comparison, 104, wrongOperand && offset == 0 ? 103 : 105)
            for lane in 0..<laneCount {
                let word = offset * laneCount + lane
                let predicate = vector ? 150 + word : comparison
                if vector { words += i(81, 6, predicate, comparison, lane) }
                words += i(169, 2, 200 + word, predicate, 51, 50)
                words += i(65, 16, 250 + word, 21, 50, 50 + word) + i(62, 250 + word, 200 + word)
            }
        }
        return words + i(253) + i(56)
    }

    @Test func spirvFloatComparisonRejectsMismatchedOperandAndPredicateTypes() throws {
        try fails(Self.floatComparisonModule(wrongResult: true))
        try fails(Self.floatComparisonModule(vector: false, wrongResult: true))
        try fails(Self.floatComparisonModule(wrongOperand: true))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl], [false, true])
func spirvFloatComparisonCompilesAllPredicates(_ format: ShaderOutputFormat, _ half: Bool) throws {
    let compiler = IndependentShaderCompiler()
    for vector in [false, true] {
        let input = ShaderInput.spirv(SPIRVFrontendTests.floatComparisonModule(vector: vector, half: half))
        let shader = try compiler.compileCompute(input, to: format)
        #expect(shader.requiresShaderF16 == half)
        guard case .text(let target, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(target == format)
        if format == .wgsl {
            _ = try compiler.parseAndValidate(.wgsl(source), capabilities: half ? [.shaderFloat16] : [])
            _ = try compiler.compileWebGPUCompute(input)
        }
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvFloatComparisonPreservesNaNsSignedZeroAndInfinityOnMetal(_ half: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.floatComparisonModule(half: half)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 192, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 192, usage: [.copyDestination, .mapRead]))
        let words: [UInt32] = half ? [0xfc000000, 0x7e007c00, 0x3c008000, 0x7c01fc00] : [
            0, 0xff800000, 0x7f800000, 0x7fc01234,
            0x80000000, 0x3f800000, 0xff800000, 0x7f800001,
        ]
        let asymmetric: [UInt32] = half ? [0x3c00fe00, 0x80010001, 0x7c010000, 0] : [
            0xffc01234, 0x3f800000, 1, 0x80000001,
            0, 0x7f800001, 0, 0,
        ]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: words),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 192), .read(buffer: read, byteCount: 192),
            .write(buffer: input, offset: 0, words: asymmetric),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 192), .read(buffer: read, byteCount: 192),
        ]).wait()
        #expect(snapshots == [[
            1, 0, 0, 0, 1, 0, 0, 1,
            0, 1, 1, 0, 0, 1, 1, 1,
            0, 1, 0, 0, 0, 1, 0, 1,
            0, 0, 1, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 1, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 1,
        ], [
            0, 0, 0, 0, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 1, 1, 1, 0, 1,
            0, 0, 1, 0, 1, 1, 1, 0,
            0, 0, 0, 1, 1, 1, 0, 1,
            0, 0, 1, 0, 1, 1, 1, 0,
        ]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
