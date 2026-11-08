import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func glslMathModule(firstOperation: [UInt32] = [80, 11, 20]) -> [UInt32] {
        var words = head + i(17, 1) + i(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0) + i(14, 0, 1)
        for instruction in [
            i(15, 5, 10, 0x6e69616d, 0), i(16, 10, 17, 1, 1, 1),
            i(71, 6, 2), i(72, 6, 0, 35, 0), i(72, 6, 1, 35, 4), i(72, 6, 2, 35, 8),
            i(72, 6, 3, 35, 12), i(72, 6, 4, 35, 16), i(72, 6, 5, 35, 20),
            i(72, 6, 6, 35, 24), i(72, 6, 7, 35, 28), i(71, 11, 34, 0), i(71, 11, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 4), i(23, 5, 3, 3),
            i(30, 6, 3, 3, 2, 2, 3, 3, 3, 2), i(32, 7, 12, 6), i(32, 8, 12, 3), i(32, 9, 12, 2), i(33, 12, 1),
            i(43, 3, 20, Float(90).bitPattern), i(43, 3, 21, 0), i(43, 3, 22, Float(1).bitPattern),
            i(43, 3, 23, Float(0.5).bitPattern), i(43, 3, 24, Float(3).bitPattern), i(43, 3, 25, Float(4).bitPattern),
            i(43, 2, 26, UInt32(bitPattern: -5)), i(44, 4, 27, 21, 22, 21, 22), i(44, 5, 28, 24, 25, 21),
            i(43, 2, 30, 0), i(43, 2, 31, 1), i(43, 2, 32, 2), i(43, 2, 33, 3),
            i(43, 2, 34, 4), i(43, 2, 35, 5), i(43, 2, 36, 6), i(43, 2, 37, 7),
            i(59, 7, 11, 12), i(54, 1, 10, 0, 12), i(248, 40),
            [UInt32(firstOperation.count + 3) << 16 | 12, 3, 41] + firstOperation,
            i(12, 3, 42, 80, 12, 41), i(12, 3, 43, 80, 22, 21), i(12, 3, 44, 80, 23, 22), i(12, 3, 45, 80, 24, 23),
            i(129, 3, 46, 43, 44), i(129, 3, 47, 46, 45), i(12, 2, 48, 80, 5, 26),
            i(12, 2, 49, 80, 55, 27), i(12, 3, 50, 80, 66, 28), i(12, 3, 51, 80, 53, 23, 32),
            i(12, 3, 58, 80, 71, 24, 22), i(12, 2, 59, 80, 74, 26),
            i(65, 8, 52, 11, 30), i(62, 52, 42), i(65, 8, 53, 11, 31), i(62, 53, 47),
            i(65, 9, 54, 11, 32), i(62, 54, 48), i(65, 9, 55, 11, 33), i(62, 55, 49),
            i(65, 8, 56, 11, 34), i(62, 56, 50), i(65, 8, 57, 11, 35), i(62, 57, 51),
            i(65, 8, 60, 11, 36), i(62, 60, 58), i(65, 9, 61, 11, 37), i(62, 61, 59), i(253), i(56),
        ] {
            words += instruction
        }
        return words
    }

    @Test func spirvGLSLMathRejectsWrongSetsArityAndOperandTypes() throws {
        try fails(Self.glslMathModule(firstOperation: [20, 11, 20]))
        try fails(Self.glslMathModule(firstOperation: [80, 11]))
        try fails(Self.glslMathModule(firstOperation: [80, 11, 30]))
        try fails(Self.glslMathModule(firstOperation: [80, 34, 20]))
        try fails(Self.glslMathModule(firstOperation: [80, 76, 20]), code: .unsupportedConstruct)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvGLSLMathTranslatesNumericPackingAndGeometry(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let result = try compiler.compileCompute(.spirv(SPIRVFrontendTests.glslMathModule()), to: format)
    guard case .text(let actual, let text) = result.artifact else { Issue.record("Expected shader text"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

#if canImport(Metal)
@Test func spirvGLSLMathPreservesSignedBitsAndFloatingResultsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.glslMathModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 32),
            .read(buffer: readback, byteCount: 32),
        ]).wait()
        let words = try #require(result.first)
        try #require(words.count == 8)
        #expect(abs(Float(bitPattern: words[0]) - 90) < 0.001)
        #expect(abs(Float(bitPattern: words[1]) - 0.54930615) < 0.00001)
        #expect(Array(words[2...]) == [5, 0xff00ff00, Float(5).bitPattern, Float(2).bitPattern, Float(-3).bitPattern, 2])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
