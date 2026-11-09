import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func conversionModule(wide: Bool, floatToUnsignedResult: UInt32 = 2, halfResult: UInt32 = 7) -> [UInt32] {
        let fields: [UInt32] = wide ? [2, 2, 2, 2, 2, 2, 5, 5] : [2, 2, 2, 2]
        var words = head + i(17, 1) + i(17, 9)
        if wide { words += i(17, 11) }
        words += i(14, 0, 1) + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        words += i(71, 12, 2) + i(71, 16, 34, 0) + i(71, 16, 33, 0)
        for field in fields.indices {
            let offset = field < 6 ? field * 4 : 24 + (field - 6) * 8
            words += i(72, 12, UInt32(field), 35, UInt32(offset))
        }
        words += i(19, 1) + i(21, 2, 32, 0) + i(21, 3, 32, 1) + i(22, 4, 32) + i(22, 7, 16)
        if wide { words += i(21, 5, 64, 0) }
        words += i(23, 8, 4, 2) + i(23, 9, 2, 2)
        words += [UInt32(fields.count + 2) << 16 | 30, 12] + fields
        words += i(32, 13, 12, 12) + i(32, 14, 12, 2) + i(33, 17, 1)
        if wide { words += i(32, 15, 12, 5) }
        for instruction in [
            i(43, 4, 18, Float(-3.75).bitPattern), i(43, 4, 19, Float(9.75).bitPattern),
            i(43, 4, 20, Float(1.0007).bitPattern), i(44, 8, 21, 18, 19),
        ] { words += instruction }
        if wide { words += i(43, 2, 22, .max) + i(43, 3, 23, .max) + i(43, 5, 24, 0x80000004, 0xffffabcd) }
        for field in fields.indices { words += i(43, 2, 25 + UInt32(field), UInt32(field)) }
        words += i(59, 13, 16, 12) + i(54, 1, 40, 0, 17) + i(248, 41)
        for instruction in [
            i(109, floatToUnsignedResult, 42, 19), i(110, 9, 43, 21),
            i(81, 2, 44, 43, 0), i(81, 2, 45, 43, 1),
            i(115, halfResult, 46, 20), i(115, 4, 47, 46), i(124, 2, 48, 47),
        ] { words += instruction }
        if wide {
            words += i(114, 2, 49, 24) + i(113, 2, 50, 24) + i(113, 5, 51, 23) + i(114, 5, 52, 22)
        }
        let values: [UInt32] = wide ? [42, 44, 45, 48, 49, 50, 51, 52] : [42, 44, 45, 48]
        for (field, value) in values.enumerated() {
            let index = UInt32(field)
            words += i(65, field < 6 ? 14 : 15, 60 + index, 16, 25 + index) + i(62, 60 + index, value)
        }
        return words + i(253) + i(56)
    }

    @Test func spirvNumericConversionsRejectInvalidSignAndWidths() throws {
        try fails(Self.conversionModule(wide: false, floatToUnsignedResult: 3))
        try fails(Self.conversionModule(wide: false, halfResult: 4))
        var words = Self.conversionModule(wide: true)
        let truncate = try #require(words.firstIndex(of: 4 << 16 | 114))
        words[truncate + 1] = 5
        try fails(words)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl], [false, true])
func spirvNumericConversionsReachTextBackends(_ format: ShaderOutputFormat, _ wide: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.conversionModule(wide: wide)), to: format)
    #expect(shader.requiresShaderInt64 == wide)
    #expect(shader.requiresShaderF16)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [wide ? 40 : 16])
    if case .text(.wgsl, let source) = shader.artifact {
        let module = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16, .shaderInt64])
        #expect(module.module.entryPoints.count == 1)
    }
}

#if canImport(Metal)
@Test func spirvNumericConversionsKeepSignednessAndWidthsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.conversionModule(wide: true)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16, .shaderInt64])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 40, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 40, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 40), .read(buffer: read, byteCount: 40),
        ]).wait()
        #expect(result == [[9, 0xfffffffd, 9, Float(1.0009765625).bitPattern, 0x80000004, 0x80000004, .max, 0, .max, .max]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
