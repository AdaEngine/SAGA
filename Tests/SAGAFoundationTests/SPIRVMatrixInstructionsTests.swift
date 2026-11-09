import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func matrixInstructionModule(transposeType: UInt32 = 7, productType: UInt32 = 5, outerType: UInt32 = 6) -> [UInt32] {
        var words = [UInt32(0x07230203), 0x00010300, 0, 256, 0] + i(17, 1) + i(14, 0, 1)
        words += i(15, 5, 18, 0x6e69616d, 0) + i(16, 18, 17, 1, 1, 1)
        for (block, count): (UInt32, UInt32) in [(9, 6), (10, 25)] {
            words += i(71, block, 2)
            for index in 0..<count { words += i(72, block, index, 35, index * 4) }
        }
        words += i(71, 16, 34, 0) + i(71, 16, 33, 0) + i(71, 17, 34, 0) + i(71, 17, 33, 1)
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 2), i(23, 5, 3, 3),
            i(24, 6, 5, 2), i(24, 7, 4, 3), i(24, 8, 5, 3),
            [UInt32(8) << 16 | 30, 9] + Array(repeating: 3, count: 6),
            [UInt32(27) << 16 | 30, 10] + Array(repeating: 3, count: 25),
            i(32, 11, 12, 9), i(32, 12, 12, 10), i(32, 13, 12, 3), i(33, 14, 1),
        ] { words += instruction }
        for index: UInt32 in 0..<25 { words += i(43, 2, 30 + index, index) }
        for (index, value) in [Float(2), 7, 8, 9, 10, 11].enumerated() {
            words += i(43, 3, 70 + UInt32(index), value.bitPattern)
        }
        words += i(44, 4, 76, 71, 72) + i(44, 5, 77, 73, 74, 75)
        words += i(59, 11, 16, 12) + i(59, 12, 17, 12) + i(54, 1, 18, 0, 14) + i(248, 19)
        for index: UInt32 in 0..<6 {
            let address = 100 + 2 * index
            words += i(65, 13, address, 16, 30 + index) + i(61, 3, address + 1, address)
        }
        for instruction in [
            i(80, 5, 112, 101, 103, 105), i(80, 5, 113, 107, 109, 111), i(80, 6, 114, 112, 113),
            i(84, transposeType, 120, 114), i(142, 4, 121, 76, 70), i(143, 6, 122, 114, 70),
            i(144, 4, 123, 77, 114), i(145, productType, 124, 114, 76), i(146, 8, 125, 114, 120),
            i(147, outerType, 126, 77, 76), i(81, 4, 127, 120, 0), i(148, 3, 128, 76, 127),
        ] { words += instruction }
        let extracts: [(UInt32, [UInt32])] = [
            (120, [0, 0]), (120, [0, 1]), (120, [1, 0]), (120, [1, 1]), (120, [2, 0]), (120, [2, 1]),
            (121, [0]), (121, [1]), (122, [1, 2]), (122, [0, 1]), (123, [0]), (123, [1]),
            (124, [0]), (124, [1]), (124, [2]), (125, [0, 0]), (125, [1, 2]), (125, [2, 2]),
            (126, [0, 0]), (126, [0, 1]), (126, [0, 2]), (126, [1, 0]), (126, [1, 1]), (126, [1, 2]),
        ]
        for (offset, extract) in extracts.enumerated() {
            let result = 130 + UInt32(offset)
            words += [UInt32(extract.1.count + 4) << 16 | 81, 3, result, extract.0] + extract.1
            words += i(65, 13, 170 + UInt32(offset), 17, 30 + UInt32(offset)) + i(62, 170 + UInt32(offset), result)
        }
        return words + i(65, 13, 194, 17, 54) + i(62, 194, 128) + i(253) + i(56)
    }

    @Test func spirvMatrixInstructionsRejectIncompatibleShapes() throws {
        try fails(Self.matrixInstructionModule(transposeType: 6))
        try fails(Self.matrixInstructionModule(productType: 4))
        try fails(Self.matrixInstructionModule(outerType: 7))
        var words = Self.matrixInstructionModule()
        let dot = try #require(words.firstIndex(of: 5 << 16 | 148))
        words[dot + 4] = 77
        try fails(words)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvMatrixInstructionsReachAllTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fixture = ShaderInput.spirv(SPIRVFrontendTests.matrixInstructionModule())
    let shader = try compiler.compileCompute(fixture, to: format)
    guard case .text(let target, let code) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    #expect(code.contains("transpose("))
    #expect(code.contains("dot("))
    if format == .hlsl { #expect(code.contains("mul(")) }
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(code))
        _ = try compiler.compileWebGPUCompute(fixture)
    }
}

#if canImport(Metal)
@Test func spirvMatrixInstructionsKeepColumnAndOperandOrderOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.matrixInstructionModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 24, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 100, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 100, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [Float(1), 2, 3, 4, 5, 6].map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 100), .read(buffer: read, byteCount: 100),
        ]).wait()
        let expected: [Float] = [1, 4, 2, 5, 3, 6, 14, 16, 12, 4, 62, 152, 39, 54, 69, 17, 36, 45, 63, 70, 77, 72, 80, 88, 39]
        #expect(snapshots == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
