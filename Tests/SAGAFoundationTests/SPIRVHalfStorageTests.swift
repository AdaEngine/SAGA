import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func halfStorageModule(capability: Bool = true) -> [UInt32] {
        var words = head + i(17, 1)
        if capability { words += i(17, 4433) }
        words += i(14, 0, 1) + i(15, 5, 20, 0x6e69616d, 0) + i(16, 20, 17, 1, 1, 1)
        for instruction in [
            i(71, 5, 2), i(72, 5, 0, 35, 0), i(72, 5, 1, 35, 2), i(71, 14, 34, 0), i(71, 14, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(22, 4, 16), i(30, 5, 4, 4),
            i(32, 6, 12, 5), i(32, 7, 12, 4), i(33, 8, 1),
            i(43, 2, 12, 0), i(43, 3, 13, 0x3f800000), i(59, 6, 14, 12), i(54, 1, 20, 0, 8), i(248, 21),
            i(65, 7, 22, 14, 12), i(61, 4, 23, 22), i(115, 3, 24, 23), i(129, 3, 25, 24, 13),
            i(115, 4, 26, 25), i(62, 22, 26), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvHalfStorageRequiresCapabilityForStorageOnlyTypes() throws {
        try fails(Self.halfStorageModule(capability: false), code: .unsupportedCapability)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvHalfStorageTranslatesStorageOnlyFloat16(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.halfStorageModule())
    let artifact = try compiler.compileCompute(input, to: format)
    #expect(artifact.requiresShaderF16)
    #expect(!artifact.requiresShaderInt16)
    #expect(artifact.storageBindings.map(\.minimumByteSize) == [4])
    if format == .wgsl { _ = try compiler.compileWebGPUCompute(input) }
}

#if canImport(Metal)
@Test func spirvHalfStorageKeepsAdjacentValueOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.halfStorageModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let buffer = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: buffer, offset: 0, words: [0xbc003e00]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): buffer], workgroups: .init(x: 1)),
            .copy(source: buffer, destination: read, byteCount: 4), .read(buffer: read, byteCount: 4),
        ]).wait()
        #expect(snapshots == [[0xbc004100]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
