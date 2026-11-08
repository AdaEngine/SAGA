import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func atomicTicketsModule(semantics: UInt32 = 0) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 11, 0x6e69616d, 0, 16), i(16, 11, 17, 64, 1, 1),
            i(71, 16, 11, 28), i(71, 5, 6, 4), i(71, 6, 2), i(72, 6, 0, 35, 0), i(72, 6, 1, 35, 512),
            i(71, 14, 34, 0), i(71, 14, 33, 0), i(71, 15, 34, 0), i(71, 15, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(23, 3, 2, 3), i(43, 2, 4, 128),
            i(28, 5, 2, 4), i(30, 6, 5, 2), i(32, 7, 12, 6), i(32, 8, 12, 2), i(32, 9, 1, 3),
            i(32, 10, 1, 2), i(33, 12, 1), i(43, 2, 20, 0), i(43, 2, 21, 1), i(43, 2, 22, 2),
            i(43, 2, 23, semantics), i(43, 2, 24, 17), i(59, 7, 14, 12), i(59, 7, 15, 12), i(59, 9, 16, 1),
            i(54, 1, 11, 0, 12), i(248, 30), i(65, 10, 31, 16, 20), i(61, 2, 32, 31),
            i(65, 8, 33, 14, 20, 20), i(234, 2, 34, 33, 21, 23, 21),
            i(65, 8, 35, 14, 20, 21), i(228, 35, 21, 23, 24), i(227, 2, 36, 35, 21, 23),
            i(65, 8, 37, 14, 20, 22), i(233, 2, 38, 37, 21, 23),
            i(65, 8, 39, 15, 20, 32), i(62, 39, 34), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvAtomicsRejectUnsupportedOrdering() throws {
        try fails(Self.atomicTicketsModule(semantics: 0x48), code: .unsupportedConstruct)
    }

    @Test func spirvAtomicsUpgradeOnlyTheAccessedStorageRoot() throws {
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(Self.atomicTicketsModule())).module
        for (index, global) in module.globalVariables.values.prefix(2).enumerated() {
            guard case .structure(let fields, _) = try module.types.value(for: global.type).kind,
                  case .array(let element, .constant(128), 4) = try module.types.value(for: fields[0].type).kind else {
                Issue.record("Expected storage array")
                return
            }
            #expect(try module.types.value(for: element).kind == (index == 0 ? .atomic(.u32) : .scalar(.u32)))
            #expect(try module.types.value(for: fields[1].type).kind == .scalar(.u32))
        }
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvAtomicsTranslateStorageLoadStoreAndRMW(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.atomicTicketsModule()), to: format)
    guard case .text(let actual, let code) = shader.artifact else { Issue.record("Expected text shader"); return }
    #expect(actual == format)
    if format == .wgsl {
        #expect(code.contains("atomicAdd(") && code.contains("atomicLoad(") && code.contains("atomicStore("))
        _ = try compiler.parseAndValidate(.wgsl(code))
    }
}

#if canImport(Metal)
@Test func spirvAtomicsReturnUniqueTicketsAcrossWorkgroupsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.atomicTicketsModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let counter = try await device.createBuffer(.init(size: 516, usage: [.storage, .copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 516, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 512, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .write(buffer: counter, offset: 0, words: [0, 0, 128]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): counter, .init(binding: 1): output],
                             workgroups: .init(x: 2)),
            .copy(source: output, destination: readback, byteCount: 512), .read(buffer: readback, byteCount: 512),
            .copy(source: counter, destination: readback, byteCount: 12), .read(buffer: readback, byteCount: 12),
        ]).wait()
        try #require(result.count == 2)
        #expect(result[0].sorted() == Array(UInt32(0)..<128))
        #expect(result[1] == [128, 17, 0])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
