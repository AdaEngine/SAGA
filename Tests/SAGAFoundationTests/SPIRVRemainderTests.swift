import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func specialRemainderModule(smodResult: UInt32 = 5, fmodRight: UInt32 = 53) -> [UInt32] {
        var words = pre + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        for block: UInt32 in [8, 9] {
            words += i(71, block, 2)
            for member: UInt32 in 0..<4 { words += i(72, block, member, 35, member * 16) }
        }
        for instruction in [
            i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(22, 4, 32),
            i(23, 5, 2, 4), i(23, 6, 3, 4), i(23, 7, 4, 4), i(30, 8, 5, 5, 7, 7), i(30, 9, 5, 7, 7, 5),
            i(32, 10, 12, 8), i(32, 11, 12, 9), i(32, 12, 12, 5), i(32, 13, 12, 7), i(33, 14, 1),
            i(43, 2, 15, 0), i(43, 2, 16, 1), i(43, 2, 17, 2), i(43, 2, 18, 3),
            i(59, 10, 20, 12), i(59, 11, 21, 12), i(54, 1, 40, 0, 14), i(248, 41),
            i(65, 12, 44, 20, 15), i(61, 5, 45, 44), i(65, 12, 46, 20, 16), i(61, 5, 47, 46), i(124, 6, 48, 47),
            i(65, 13, 50, 20, 17), i(61, 7, 51, 50), i(65, 13, 52, 20, 18), i(61, 7, 53, 52),
            i(139, smodResult, 54, 45, 48), i(140, 7, 55, 51, 53), i(141, 7, 56, 51, fmodRight),
            i(81, 2, 57, 45, 0), i(81, 3, 58, 48, 0), i(139, 2, 59, 57, 58),
            i(81, 4, 60, 51, 0), i(81, 4, 61, 53, 0), i(140, 4, 62, 60, 61), i(141, 4, 63, 60, 61),
            i(124, 2, 64, 62), i(124, 2, 65, 63), i(80, 5, 66, 59, 64, 65, 15),
            i(65, 12, 70, 21, 15), i(62, 70, 54), i(65, 13, 71, 21, 16), i(62, 71, 55),
            i(65, 13, 72, 21, 17), i(62, 72, 56), i(65, 12, 73, 21, 18), i(62, 73, 66), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvRemainderRejectsResultAndOperandMismatches() throws {
        try fails(Self.specialRemainderModule(smodResult: 7))
        try fails(Self.specialRemainderModule(fmodRight: 45))
        try fails(Self.specialRemainderModule(fmodRight: 15))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvRemainderTranslatesAllTextTargets(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fixture = ShaderInput.spirv(SPIRVFrontendTests.specialRemainderModule())
    let shader = try compiler.compileCompute(fixture, to: format)
    guard case .text(let target, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(fixture)
    }
}

#if canImport(Metal)
@Test func spirvRemainderPreservesOpcodeSignOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.specialRemainderModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let inputs = [Int32(5), 5, -5, -5, 3, -3, 3, -3].map { UInt32(bitPattern: $0) }
            + [Float(-5.5), 5.5, -5.5, 5.5, 3, -3, -3, 3].map(\.bitPattern)
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: inputs),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        ]).wait()
        let vectorRemainders = [Float(-2.5), 2.5, -2.5, 2.5, 0.5, -0.5, -2.5, 2.5].map(\.bitPattern)
        let expected: [UInt32] = [2, .max, 1, .max - 1] + vectorRemainders
            + [2, Float(-2.5).bitPattern, Float(0.5).bitPattern, 0]
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
