import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func bitInstructionsModule(countOperand: UInt32 = 48) -> [UInt32] {
        var words = pre + i(15, 5, 30, 0x6e69616d, 0) + i(16, 30, 17, 1, 1, 1)
        words += i(71, 6, 2) + i(72, 6, 0, 35, 0) + i(72, 6, 1, 35, 16) + i(71, 7, 2)
        for field: UInt32 in 0..<6 { words += i(72, 7, field, 35, field * 16) }
        for instruction in [
            i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(23, 4, 2, 4), i(23, 5, 3, 4),
            i(30, 6, 4, 4), i(30, 7, 4, 4, 4, 4, 4, 4), i(32, 8, 12, 6), i(32, 9, 12, 7),
            i(32, 10, 12, 4), i(33, 11, 1),
        ] { words += instruction }
        for field: UInt32 in 0..<6 { words += i(43, 2, 12 + field, field) }
        for instruction in [
            i(44, 4, 18, 17, 17, 17, 17), i(59, 8, 20, 12), i(59, 9, 21, 12), i(54, 1, 30, 0, 11), i(248, 31),
            i(65, 10, 40, 20, 12), i(61, 4, 41, 40), i(65, 10, 42, 20, 13), i(61, 4, 43, 42),
            i(81, 2, 44, 43, 0), i(124, 3, 45, 44), i(81, 2, 48, 43, 1), i(81, 2, 49, 43, 2),
            i(81, 2, 50, 43, 3), i(201, 4, 51, 41, 18, 45, countOperand), i(202, 4, 52, 41, 45, 48),
            i(124, 5, 53, 41), i(203, 5, 54, 53, 45, 48), i(124, 4, 55, 54), i(204, 4, 56, 41),
            i(205, 5, 57, 41), i(124, 4, 58, 57), i(201, 4, 59, 41, 18, 49, 50),
        ] { words += instruction }
        for (index, value) in [UInt32(51), 52, 55, 56, 58, 59].enumerated() {
            let field = UInt32(index)
            words += i(65, 10, 70 + field, 21, 12 + field) + i(62, 70 + field, value)
        }
        return words + i(253) + i(56)
    }

    static func wideBitInstructionsModule() -> [UInt32] {
        var words = head + i(17, 1) + i(17, 11) + i(14, 0, 1) + i(15, 5, 30, 0x6e69616d, 0)
        words += i(16, 30, 17, 1, 1, 1) + i(71, 5, 2) + i(71, 20, 34, 0) + i(71, 20, 33, 0)
        for (member, offset) in [UInt32(0), 8, 16, 24, 28].enumerated() {
            words += i(72, 5, UInt32(member), 35, offset)
        }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 64, 0), i(21, 4, 64, 1), i(30, 5, 3, 3, 3, 2, 2),
            i(32, 6, 12, 5), i(32, 7, 12, 3), i(32, 8, 12, 2), i(33, 9, 1),
            i(43, 2, 10, 0), i(43, 2, 11, 1), i(43, 2, 12, 2), i(43, 2, 13, 3), i(43, 2, 14, 4),
            i(43, 3, 15, 0xf0, 0x80000000), i(43, 3, 16, 4, 1), i(43, 4, 17, 4, 0),
            i(59, 6, 20, 12), i(54, 1, 30, 0, 9), i(248, 31),
            i(202, 3, 40, 15, 17, 14), i(203, 3, 41, 15, 16, 14), i(204, 3, 42, 15), i(205, 2, 43, 15),
            i(65, 7, 50, 20, 10), i(62, 50, 40), i(65, 7, 51, 20, 11), i(62, 51, 41),
            i(65, 7, 52, 20, 12), i(62, 52, 42), i(65, 8, 53, 20, 13), i(62, 53, 43),
            i(65, 8, 54, 20, 14), i(62, 54, 10), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvBitInstructionsRejectVectorRangeAndMismatchedBase() throws {
        try fails(Self.bitInstructionsModule(countOperand: 41))
        var words = Self.bitInstructionsModule()
        let extract = try #require(words.firstIndex(of: 6 << 16 | 202))
        words[extract + 1] = 5
        try fails(words)
    }

    @Test func spirvBitInstructionsNormalizeSixteenBitOperandsInIR() throws {
        var words = Self.head + Self.i(17, 1) + Self.i(17, 22) + Self.i(14, 0, 1)
        for instruction in [
            Self.i(15, 5, 20, 0x6e69616d, 0), Self.i(16, 20, 17, 1, 1, 1),
            Self.i(19, 1), Self.i(21, 2, 16, 0), Self.i(21, 3, 32, 0), Self.i(33, 4, 1),
            Self.i(43, 2, 7, 0x8000), Self.i(43, 3, 8, 15), Self.i(43, 3, 9, 1),
            Self.i(54, 1, 20, 0, 4), Self.i(248, 21), Self.i(202, 2, 10, 7, 8, 9),
            Self.i(204, 2, 11, 7), Self.i(205, 3, 12, 7), Self.i(195, 2, 13, 7, 9), Self.i(253), Self.i(56),
        ] { words += instruction }
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(words), capabilities: [.shaderInt16]).module
        let function = try module.functions.value(for: module.entryPoints[0].function)
        let types = try function.expressions.values.compactMap { expression -> IRTypeKind? in
            guard case .math = expression.kind else { return nil }
            return try module.types.value(for: expression.type).kind
        }
        #expect(types == [.scalar(.i32), .scalar(.u32), .scalar(.u32)])
        let compiler = IndependentShaderCompiler()
        for format: ShaderOutputFormat in [.msl, .hlsl, .glsl, .wgsl] {
            let shader = try compiler.compileCompute(.spirv(words), to: format)
            #expect(shader.requiresShaderInt16)
        }
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl], [false, true])
func spirvBitInstructionsTranslateAllTextBackends(_ format: ShaderOutputFormat, _ wide: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(wide ? SPIRVFrontendTests.wideBitInstructionsModule() : SPIRVFrontendTests.bitInstructionsModule())
    let artifact = try compiler.compileCompute(input, to: format)
    #expect(artifact.requiresShaderInt64 == wide)
    guard case .text(let target, let source) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(target == format)
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: wide ? [.shaderInt64] : [])
        if !wide { _ = try compiler.compileWebGPUCompute(input) }
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvBitInstructionsKeepOpcodeSignsAndWideCountsOnMetal(_ wide: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let module = wide ? SPIRVFrontendTests.wideBitInstructionsModule() : SPIRVFrontendTests.bitInstructionsModule()
    let artifact = try compiler.compileCompute(.spirv(module), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: wide ? [.shaderInt64] : [])
    do {
        let pipeline = try await device.createComputePipeline(artifact)
        let byteCount: UInt64 = wide ? 32 : 96
        let output = try await device.createBuffer(.init(size: byteCount, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: byteCount, usage: [.copyDestination, .mapRead]))
        let inputWords: [UInt32] = [0xa5f7f0f0, 0xf0, 0xfffffff0, 0x70]
        var commands: [BufferCommand] = []
        var bindings: [ComputeBindingSlot: ResourceID] = [.init(binding: wide ? 0 : 1): output]
        if !wide {
            let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
            bindings[.init(binding: 0)] = input
            commands.append(.write(buffer: input, offset: 0, words: inputWords + [4, 4, 32, 0]))
        }
        commands += [
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: byteCount), .read(buffer: read, byteCount: byteCount),
        ]
        let reversed = inputWords.map { value in (0..<32).reduce(UInt32(0)) { ($0 << 1) | ((value >> $1) & 1) } }
        let expected: [UInt32] = wide ? [.max, .max, 0, 0, 1, 0x0f000000, 5, 0] :
            [0xa5f7f050, 0x50, 0xffffff50, 0x50, .max, .max, .max, 7, 15, 15, 15, 7]
            + reversed + inputWords.map { UInt32($0.nonzeroBitCount) } + inputWords
        #expect(try await device.submitCommands(commands).wait() == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
