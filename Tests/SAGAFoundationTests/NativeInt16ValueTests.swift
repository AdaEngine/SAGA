import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func nativeInt16ValueModule(half: Bool = false) -> [UInt32] {
        var words: [UInt32] = [0x07230203, 0x00010300, 0, 256, 0]
        words += i(17, 1) + i(17, 22)
        if half { words += i(17, 9) }
        words += i(14, 0, 1) + i(15, 5, 60, 0x6e69616d, 0) + i(16, 60, 17, 1, 1, 1)
        let fieldCount: UInt32 = half ? 16 : 14
        for (block, count): (UInt32, UInt32) in [(17, 4), (18, fieldCount)] {
            words += i(71, block, 2)
            for index in 0..<count { words += i(72, block, index, 35, index * 4) }
        }
        words += i(71, 23, 34, 0) + i(71, 23, 33, 0) + i(71, 24, 34, 0) + i(71, 24, 33, 1)
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(21, 4, 16, 0), i(21, 5, 16, 1),
            i(23, 6, 4, 2), i(23, 7, 5, 2), i(20, 8), i(23, 9, 8, 2), i(22, 10, 32),
            i(23, 11, 2, 2), i(23, 12, 3, 2), i(30, 17, 2, 2, 2, 2),
            [UInt32(fieldCount + 2) << 16 | 30, 18] + Array(repeating: 2, count: Int(fieldCount)),
            i(32, 19, 12, 17), i(32, 20, 12, 18), i(32, 21, 12, 2), i(33, 22, 1),
            i(43, 4, 25, 1), i(43, 4, 26, 5), i(43, 4, 27, 0xffff), i(43, 5, 28, 0xfffffffd),
            i(44, 6, 29, 25, 27), i(46, 7, 30),
        ] { words += instruction }
        if half { words += i(22, 13, 16) + i(23, 14, 13, 2) + i(43, 4, 31, 0x3c00) }
        for index in 0..<fieldCount { words += i(43, 2, 40 + index, index) }
        words += i(59, 19, 23, 12) + i(59, 20, 24, 12) + i(54, 1, 60, 0, 22) + i(248, 61)
        for index: UInt32 in 0..<4 {
            let address = 62 + index * 2
            words += i(65, 21, address, 23, 40 + index) + i(61, 2, address + 1, address)
        }
        for instruction in [
            i(113, 4, 70, 63), i(113, 4, 71, 65), i(124, 5, 72, 71), i(124, 10, 73, 67), i(113, 4, 74, 69),
            i(128, 4, 80, 70, 25), i(132, 4, 81, 70, 70), i(126, 5, 82, 72),
            i(135, 5, 83, 72, 28), i(138, 5, 84, 72, 28), i(130, 4, 85, 70, 26), i(200, 4, 86, 70),
            i(110, 5, 87, 73), i(111, 10, 88, 72), i(124, 2, 89, 88),
            i(80, 6, 90, 70, 74), i(128, 6, 91, 90, 29), i(113, 11, 92, 91),
            i(81, 2, 93, 92, 0), i(81, 2, 94, 92, 1), i(124, 7, 95, 90), i(126, 7, 96, 95),
            i(114, 12, 97, 96), i(124, 11, 98, 97), i(81, 2, 99, 98, 1),
            i(177, 9, 100, 95, 30), i(81, 8, 101, 100, 0), i(81, 8, 102, 100, 1),
            i(169, 2, 103, 101, 41, 40), i(169, 2, 104, 102, 41, 40),
        ] { words += instruction }
        var results: [UInt32] = []
        for (index, value) in [UInt32(80), 81, 82, 83, 84, 85, 86, 87].enumerated() {
            let result = 110 + UInt32(index)
            if [82, 83, 84, 87].contains(value) {
                words += i(114, 3, result + 10, value) + i(124, 2, result, result + 10)
            } else {
                words += i(113, 2, result, value)
            }
            results.append(result)
        }
        results += [89, 93, 94, 99, 103, 104]
        if half {
            words += i(80, 6, 130, 71, 31) + i(124, 14, 131, 130)
            words += i(81, 13, 132, 131, 0) + i(124, 4, 133, 132) + i(113, 2, 134, 133)
            words += i(81, 13, 135, 131, 1) + i(115, 10, 136, 135) + i(124, 2, 137, 136)
            results += [134, 137]
        }
        for (index, value) in results.enumerated() {
            let field = UInt32(index)
            words += i(65, 21, 140 + field, 24, 40 + field) + i(62, 140 + field, value)
        }
        return words + i(253) + i(56)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func nativeInt16ValuesTranslateWithoutPromotingDeclarations(_ format: ShaderOutputFormat, _ half: Bool) throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.nativeInt16ValueModule(half: half))
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.requiresShaderF16 == half)
    guard case .text(let target, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    switch format {
    case .msl: #expect(source.contains("short2") && source.contains("ushort2"))
    case .hlsl: #expect(source.contains("vector<int16_t, 2>") && source.contains("vector<uint16_t, 2>"))
    case .glsl: #expect(source.contains("i16vec2") && source.contains("u16vec2"))
    case .wgsl:
        #expect(source.contains("vec2<i16>") && source.contains("vec2<u16>"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: half ? [.shaderInt16, .shaderFloat16] : [.shaderInt16])
    default: throw ShaderFailure.incompatibleArtifact
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func nativeInt16ValuesPreserveWrapAndBitPatternsOnMetal(_ half: Bool) async throws {
    let artifact = try IndependentShaderCompiler().compileCompute(
        .spirv(SPIRVFrontendTests.nativeInt16ValueModule(half: half)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(artifact)
        let byteCount: UInt64 = half ? 64 : 56
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: byteCount, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: byteCount, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0x1234ffff, 0x8000, Float(-12345.75).bitPattern, 7]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: byteCount), .read(buffer: read, byteCount: byteCount),
        ]).wait()
        var expected: [UInt32] = [0, 1, 0xffff8000, 10922, 0xfffffffe, 65530, 0, 0xffffcfc7,
                                  0xc7000000, 0, 6, 0xfffffff9, 1, 0]
        if half { expected += [0x8000, 0x3f800000] }
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
