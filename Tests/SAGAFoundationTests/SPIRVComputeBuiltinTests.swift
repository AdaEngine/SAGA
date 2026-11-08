import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func computeBuiltinsModule(varyingBarrier: Bool = false, omitGroup: Bool = false) -> [UInt32] {
        var words = pre
        let interfaces: [UInt32] = omitGroup ? [15, 16, 18, 19] : [15, 16, 17, 18, 19]
        words += [UInt32(interfaces.count + 5) << 16 | 15, 5, 12, 0x6e69616d, 0] + interfaces
        for instruction in [
            i(16, 12, 17, 1, 1, 1),
            i(71, 15, 11, 28), i(71, 16, 11, 27), i(71, 17, 11, 26), i(71, 18, 11, 24), i(71, 19, 11, 29),
            i(71, 28, 11, 25), i(71, 5, 6, 4), i(71, 6, 2), i(72, 6, 0, 35, 0), i(71, 14, 34, 0), i(71, 14, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(20, 3), i(23, 4, 2, 3),
            i(43, 2, 20, 0), i(43, 2, 21, 1), i(43, 2, 22, 2), i(43, 2, 23, 4),
            i(43, 2, 24, 16), i(43, 2, 25, 64), i(43, 2, 26, 0x108), i(43, 2, 27, 32),
            i(44, 4, 28, 22, 21, 21), i(28, 5, 2, 27), i(30, 6, 5),
            i(32, 7, 1, 4), i(32, 8, 1, 2), i(32, 9, 12, 6), i(32, 10, 12, 2), i(33, 11, 1),
            i(59, 9, 14, 12), i(59, 7, 15, 1), i(59, 7, 16, 1), i(59, 7, 17, 1), i(59, 7, 18, 1), i(59, 8, 19, 1),
            i(54, 1, 12, 0, 11), i(248, 30), i(57, 1, 31, 13), i(253), i(56),
            i(54, 1, 13, 0, 11), i(248, 40),
            i(65, 8, 41, 15, 20), i(61, 2, 42, 41), i(65, 8, 43, 16, 20), i(61, 2, 44, 43),
            i(65, 8, 45, 17, 20), i(61, 2, 46, 45), i(65, 8, 47, 18, 20), i(61, 2, 48, 47), i(61, 2, 49, 19),
            i(170, 3, 50, varyingBarrier ? 49 : 46, 20), i(247, 53, 0), i(250, 50, 52, 53),
            i(248, 52), i(224, 22, 22, 26), i(249, 53), i(248, 53),
            i(132, 2, 54, 49, 23), i(132, 2, 55, 46, 24), i(132, 2, 56, 48, 25),
            i(128, 2, 57, 44, 54), i(128, 2, 58, 57, 55), i(128, 2, 59, 58, 56),
            i(65, 10, 60, 14, 20, 42), i(62, 60, 59), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvComputeBuiltinsRejectUnlistedInputsAndVaryingBarriers() throws {
        try fails(Self.computeBuiltinsModule(omitGroup: true))
        do {
            _ = try IndependentShaderCompiler().parseAndValidate(.spirv(Self.computeBuiltinsModule(varyingBarrier: true)))
            Issue.record("Accepted divergent workgroup barrier")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.first?.code == .nonUniformControlFlow)
        }
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvComputeBuiltinsTranslateWorkgroupUniformHelpers(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.computeBuiltinsModule()), to: format)
    #expect(shader.workgroupSize == .init(x: 2, y: 1, z: 1))
    guard case .text(let actual, let code) = shader.artifact else { Issue.record("Expected text shader"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code)) }
}

#if canImport(Metal)
@Test func spirvComputeBuiltinsExecuteAcrossGroupsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.computeBuiltinsModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 24, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 3)),
            .copy(source: output, destination: readback, byteCount: 24), .read(buffer: readback, byteCount: 24),
        ]).wait()
        #expect(result == [[192, 197, 208, 213, 224, 229]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
