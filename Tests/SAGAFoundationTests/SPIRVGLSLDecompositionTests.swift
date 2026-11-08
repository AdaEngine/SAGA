import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func glslDecompositionModule(invalidExponent: Bool = false) -> [UInt32] {
        var words = head + i(17, 1) + i(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0) + i(14, 0, 1)
        words += i(15, 5, 10, 0x6e69616d, 0) + i(16, 10, 17, 1, 1, 1)
        words += i(6, 5, 1, 0x74726170, 0) + i(71, 11, 34, 0) + i(71, 11, 33, 0) + i(71, 7, 2)
        for index: UInt32 in 0..<12 { words += i(72, 7, index, 35, index * 4) }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 3), i(23, 13, 2, 3),
            i(30, 5, 4, 4), i(30, 6, 4, invalidExponent ? 4 : 13),
            i(30, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2),
            i(32, 8, 12, 7), i(32, 9, 12, 3), i(32, 12, 12, 2), i(33, 14, 1),
            i(43, 3, 20, Float(-3.25).bitPattern), i(43, 3, 21, Float(0.125).bitPattern), i(43, 3, 22, Float(8.5).bitPattern),
            i(44, 4, 23, 20, 21, 22),
        ] {
            words += instruction
        }
        for index: UInt32 in 0..<12 { words += i(43, 2, 24 + index, index) }
        words += i(59, 8, 11, 12) + i(54, 1, 10, 0, 14) + i(248, 40)
        words += i(12, 5, 41, 80, 36, 23) + i(12, 6, 42, 80, 52, 23) + i(12, 5, 43, 80, 36, 23)
        for index: UInt32 in 0..<12 {
            let type: UInt32 = index < 9 ? 3 : 2
            let result: UInt32 = index < 6 ? 41 : 42
            words += i(81, type, 44 + index, result, (index / 3) % 2, index % 3)
            words += i(65, index < 9 ? 9 : 12, 56 + index, 11, 24 + index)
            words += i(62, 56 + index, 44 + index)
        }
        return words + i(253) + i(56)
    }

    @Test func spirvGLSLDecompositionKeepsValueLayoutAndReusesCanonicalTypes() throws {
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(Self.glslDecompositionModule())).module
        #expect(module.specialTypes.predeclared.count == 2)
        let pair = try #require(module.types.values.first { type in
            if case .structure(let members, _) = type.kind { return members.last?.name == "part" }
            return false
        })
        guard case .structure(let members, let span) = pair.kind else { Issue.record("Expected result structure"); return }
        #expect(members.map(\.offset) == [0, 16])
        #expect(span == 32)
    }

    @Test func spirvGLSLDecompositionRejectsFloatingFrexpExponent() throws {
        try fails(Self.glslDecompositionModule(invalidExponent: true))
    }

    @Test func spirvGLSLDecompositionAlignsHalfSignificandsAndIntegerExponents() throws {
        var words = Self.head
        for instruction in [
            Self.i(17, 1), Self.i(17, 9), Self.i(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0), Self.i(14, 0, 1),
            Self.i(15, 5, 10, 0x6e69616d, 0), Self.i(16, 10, 17, 1, 1, 1),
            Self.i(19, 1), Self.i(21, 2, 32, 1), Self.i(22, 3, 16), Self.i(30, 4, 3, 2),
            Self.i(33, 5, 1), Self.i(43, 3, 6, 0xb000), Self.i(54, 1, 10, 0, 5), Self.i(248, 20),
            Self.i(12, 4, 21, 80, 52, 6), Self.i(253), Self.i(56),
        ] {
            words += instruction
        }
        let compiler = IndependentShaderCompiler()
        let capabilities = ShaderCapabilities.referenceDefault.union(.shaderFloat16)
        let module = try compiler.parseAndValidate(.spirv(words), capabilities: capabilities).module
        let binding = try #require(module.specialTypes.predeclared.first)
        let result = try module.types.value(for: binding.type)
        guard case .structure(let members, let span) = result.kind else { Issue.record("Expected result structure"); return }
        #expect(members.map(\.offset) == [0, 4])
        #expect(span == 8)
        let shader = try compiler.compileCompute(.spirv(words), to: .wgsl)
        guard case .text(.wgsl, let text) = shader.artifact else { Issue.record("Expected WGSL"); return }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: capabilities)
    }

    @Test func spirvGLSLDecompositionRequiresExplicitOffsetsForNestedBufferTypes() throws {
        var words = Self.pre
        for instruction in [
            Self.i(71, 3, 2), Self.i(72, 3, 0, 35, 0), Self.i(71, 5, 34, 0), Self.i(71, 5, 33, 0),
            Self.i(21, 1, 32, 0), Self.i(30, 2, 1), Self.i(30, 3, 2), Self.i(32, 4, 12, 3), Self.i(59, 4, 5, 12),
        ] {
            words += instruction
        }
        try fails(words)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvGLSLDecompositionTranslatesStructResults(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileCompute(.spirv(SPIRVFrontendTests.glslDecompositionModule()), to: format)
    guard case .text(let actual, let text) = artifact.artifact else { Issue.record("Expected shader text"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

#if canImport(Metal)
@Test func spirvGLSLDecompositionPreservesSignedExponentsAndFractionsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.glslDecompositionModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 48),
            .read(buffer: readback, byteCount: 48),
        ]).wait()
        let expected: [Float] = [-0.25, 0.125, 0.5, -3, 0, 8, -0.8125, 0.5, 0.53125]
        #expect(result == [expected.map(\.bitPattern) + [2, UInt32(bitPattern: -2), 4]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
