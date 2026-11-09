import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func explicitTailPaddingModule(narrow: Bool = true, trailingOffset: UInt32? = nil) -> [UInt32] {
        let scalar: UInt32 = narrow ? 3 : 2
        let width: UInt32 = narrow ? 2 : 4
        let stride = 4 * width
        var words = head + i(17, 1)
        if narrow { words += i(17, 22) + i(17, 4433) }
        words += i(14, 0, 1) + i(15, 5, 20, 0x6e69616d, 0) + i(16, 20, 17, 1, 1, 1)
        for instruction in [
            i(72, 6, 0, 35, 0), i(72, 6, 1, 35, width * 2), i(71, 7, 6, stride),
            i(71, 8, 2), i(72, 8, 0, 35, 0), i(72, 8, 1, 35, trailingOffset ?? stride * 2),
            i(71, 17, 34, 0), i(71, 17, 33, 0), i(19, 1), i(21, 2, 32, 0),
        ] { words += instruction }
        if narrow { words += i(21, 3, 16, 0) }
        for instruction in [
            i(23, 5, scalar, 2), i(30, 6, 5, scalar), i(43, 2, 11, 2), i(28, 7, 6, 11),
            i(30, 8, 7, 2), i(32, 9, 12, 8), i(32, 10, 12, scalar), i(33, 12, 1),
            i(43, 2, 13, 0), i(43, 2, 14, 1), i(43, scalar, 15, 1),
            i(59, 9, 17, 12), i(54, 1, 20, 0, 12), i(248, 21), i(65, 10, 22, 17, 13, 14, 14),
            i(61, scalar, 23, 22), i(128, scalar, 24, 23, 15), i(62, 22, 24), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvExplicitStructPaddingRejectsSpanOverflow() throws {
        try fails(Self.explicitTailPaddingModule(narrow: false, trailingOffset: 0x7fff_fffe))
    }
}

@Test(arguments: [true, false])
func spirvExplicitStructPaddingPreservesMemberOffsetsAndBindingSpan(_ narrow: Bool) throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.explicitTailPaddingModule(narrow: narrow))
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input, capabilities: .referenceDefault.union(.shaderInt16)).module
    let root = try #require(module.globalVariables.values.first)
    guard case .structure(let fields, let span) = try module.types.value(for: root.type).kind,
          case .array(let element, .constant(2), let stride) = try module.types.value(for: fields[0].type).kind,
          case .structure(let nested, let nestedSpan) = try module.types.value(for: element).kind else {
        Issue.record("Expected explicit structure with nested array")
        return
    }
    #expect(nested.map(\.offset) == [0, narrow ? 4 : 8])
    #expect(fields.map(\.offset) == [0, narrow ? 16 : 32])
    #expect(nestedSpan == (narrow ? 8 : 16))
    #expect(stride == nestedSpan)
    #expect(span == (narrow ? 20 : 40))
    let shader = try compiler.compileCompute(input, to: .msl)
    #expect(shader.storageBindings.first?.minimumByteSize == (narrow ? 20 : 40))
}

#if canImport(Metal)
@Test func spirvExplicitStructPaddingPreservesPaddingAndAdjacentElementOnMetal() async throws {
    let source = ShaderInput.spirv(SPIRVFrontendTests.explicitTailPaddingModule())
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let storage = try await device.createBuffer(.init(size: 20, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 20, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: storage, offset: 0, words: [0x0014000a, 0xabcd1111, 0x0028001e, 0xdef02222, 0xcafe0102]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): storage], workgroups: .init(x: 1)),
            .copy(source: storage, destination: read, byteCount: 20), .read(buffer: read, byteCount: 20),
        ]).wait()
        #expect(snapshots == [[0x0014000a, 0xabcd1111, 0x0028001e, 0xdef02223, 0xcafe0102]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
