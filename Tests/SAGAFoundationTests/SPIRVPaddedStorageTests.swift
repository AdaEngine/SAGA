import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func paddedStorageModule() -> [UInt32] {
        var words = pre + i(15, 5, 30, 0x6e69616d, 0) + i(16, 30, 17, 1, 1, 1)
        for instruction in [
            i(71, 5, 6, 16), i(71, 6, 6, 16), i(71, 8, 2), i(71, 9, 2),
            i(72, 8, 0, 35, 0), i(72, 8, 1, 35, 32), i(72, 8, 2, 35, 48),
            i(72, 9, 0, 35, 0), i(72, 9, 1, 35, 4), i(72, 9, 2, 35, 16),
            i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(43, 2, 16, 0), i(43, 2, 17, 1), i(43, 2, 18, 2),
            i(29, 5, 2), i(28, 6, 2, 18), i(30, 8, 6, 2, 5), i(30, 9, 2, 2, 6),
            i(32, 10, 12, 8), i(32, 11, 12, 9), i(32, 12, 12, 2), i(32, 13, 12, 6), i(33, 14, 1),
            i(59, 10, 20, 12), i(59, 11, 21, 12), i(54, 1, 30, 0, 14), i(248, 31),
            i(65, 13, 32, 20, 16), i(61, 6, 33, 32), i(65, 13, 34, 21, 18), i(62, 34, 33),
            i(68, 2, 35, 20, 2), i(65, 12, 36, 21, 16), i(62, 36, 35),
            i(65, 12, 37, 20, 18, 17), i(61, 2, 38, 37), i(65, 12, 39, 21, 17), i(62, 39, 38), i(253), i(56),
        ] { words += instruction }
        return words
    }

    static func paddedInt16StorageModule() -> [UInt32] {
        var words = head + i(17, 1) + i(17, 22) + i(17, 4433) + i(14, 0, 1)
        words += i(15, 5, 30, 0x6e69616d, 0) + i(16, 30, 17, 1, 1, 1)
        for instruction in [
            i(71, 5, 6, 6), i(71, 6, 2), i(71, 7, 2), i(72, 6, 0, 35, 0), i(72, 6, 1, 35, 4),
            i(72, 7, 0, 35, 0), i(72, 7, 1, 35, 4), i(71, 20, 34, 0), i(71, 20, 33, 0), i(71, 21, 34, 0), i(71, 21, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 16, 1), i(21, 4, 32, 1), i(29, 5, 3), i(30, 6, 2, 5), i(30, 7, 4, 2),
            i(32, 8, 12, 6), i(32, 9, 12, 7), i(32, 10, 12, 3), i(32, 11, 12, 4), i(32, 12, 12, 2), i(33, 13, 1),
            i(43, 2, 16, 0), i(43, 2, 17, 1), i(43, 3, 19, .max),
            i(59, 8, 20, 12), i(59, 9, 21, 12), i(54, 1, 30, 0, 13), i(248, 31),
            i(65, 10, 32, 20, 17, 17), i(61, 3, 33, 32), i(114, 4, 34, 33), i(62, 32, 19),
            i(65, 11, 35, 21, 16), i(62, 35, 34), i(68, 2, 36, 20, 1), i(65, 12, 37, 21, 17), i(62, 37, 36), i(253), i(56),
        ] { words += instruction }
        return words
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvPaddedStorageKeepsDeclaredStrideAcrossTextTargets(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fixture = ShaderInput.spirv(SPIRVFrontendTests.paddedStorageModule())
    let shader = try compiler.compileCompute(fixture, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64, 48])
    #expect(shader.storageBindings[0].runtimeArrayOffset == 48)
    #expect(shader.storageBindings[0].runtimeArrayStride == 16)
    guard case .text(_, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    if format == .wgsl {
        #expect(source.contains("@size(16) saga_value"))
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(fixture)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func spirvPaddedStoragePreservesSixteenBitElementStride(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.paddedInt16StorageModule()), to: format)
    #expect(shader.requiresShaderInt16)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [12, 8])
    #expect(shader.storageBindings[0].runtimeArrayStride == 6)
    guard case .text(_, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    if format == .wgsl {
        #expect(source.contains("@size(6) saga_value: i16"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderInt16])
    }
    if format == .glsl { #expect(source.contains("uint16_t saga_pad_0[2]")) }
}

#if canImport(Metal)
@Test func spirvPaddedStorageCopiesOnlyValuesAndUsesBindingRangeOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.paddedStorageModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 80, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
        let inputWords: [UInt32] = [11, 1, 2, 3, 22, 4, 5, 6, 7, 8, 9, 10, 33, 11, 12, 13, 44, 14, 15, 16]
        let sentinel = UInt32(0xdeadbeef)
        var commands: [BufferCommand] = [.write(buffer: input, offset: 0, words: inputWords),
            .write(buffer: output, offset: 0, words: Array(repeating: sentinel, count: 12))]
        for size: UInt64 in [80, 64] {
            commands += [
                .dispatchComputeRanges(pipeline: pipeline,
                    bindings: [.init(binding: 0): .init(buffer: input, size: size), .init(binding: 1): .init(buffer: output)],
                    workgroups: .init(x: 1)),
                .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
            ]
        }
        let tail: [UInt32] = [sentinel, sentinel, 11, sentinel, sentinel, sentinel, 22, sentinel, sentinel, sentinel]
        #expect(try await device.submitCommands(commands).wait() == [[2, 44] + tail, [1, 0] + tail])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func spirvPaddedStoragePreservesSixteenBitNeighborsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.paddedInt16StorageModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0x01234567, 0xabcd0007, 0xfff8cdef, 0xab12ef34]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output], workgroups: .init(x: 1)),
            .copy(source: input, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
            .copy(source: output, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[0x01234567, 0xabcd0007, 0xffffcdef, 0xab12ef34], [0xfffffff8, 2]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
