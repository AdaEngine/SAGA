import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    enum StorageImageFailure: CaseIterable, Sendable {
        case scalarCoordinate, writeReadOnly, readWriteOnly, unknownFormat, mismatchedFormat
        case missingImageQueryCapability, omittedInterface

        var offendingOpcode: UInt32 {
            switch self {
            case .scalarCoordinate, .readWriteOnly: 98
            case .writeReadOnly: 99
            case .unknownFormat, .mismatchedFormat: 25
            case .missingImageQueryCapability: 104
            case .omittedInterface: 15
            }
        }
    }

    static func storageImageModule(readWrite: Bool = false, failure: StorageImageFailure? = nil) -> [UInt32] {
        let imageFormat: UInt32 = failure == .unknownFormat ? 0 : failure == .mismatchedFormat ? 3 : 33
        var words = head + i(17, 1)
        if failure != .missingImageQueryCapability { words += i(17, 50) }
        words += i(14, 0, 1)
        if failure == .omittedInterface {
            words[1] = 0x00010400
            words += i(15, 5, 40, 0x6e69616d, 0, 15)
        } else {
            words += i(15, 5, 40, 0x6e69616d, 0)
        }
        words += i(16, 40, 17, 1, 1, 1)
        words += i(71, 15, 34, 0) + i(71, 15, 33, 0) + i(71, 15, 24)
        words += i(71, 16, 34, 0) + i(71, 16, 33, 1)
        if !readWrite { words += i(71, 16, 25) }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(23, 4, 3, 2), i(23, 5, 2, 4),
            i(25, 6, 2, 1, 0, 0, 0, 2, imageFormat), i(32, 7, 0, 6), i(33, 8, 1),
            i(43, 3, 20, 0), i(43, 3, 21, 1), i(43, 3, 22, 100), i(43, 2, 23, 0), i(44, 4, 24, 21, 20),
            i(43, 3, 25, 0xffffffff), i(44, 4, 26, 25, 20),
            i(59, 7, 15, 0), i(59, 7, 16, 0), i(54, 1, 40, 0, 8), i(248, 41),
            i(61, 6, 42, 15), i(61, 6, 43, 16),
            i(98, 5, 44, failure == .readWriteOnly ? 43 : 42, failure == .scalarCoordinate ? 21 : 24),
            i(104, 4, 45, 43), i(81, 3, 46, 45, 0), i(81, 3, 47, 45, 1),
            i(132, 3, 48, 46, 22), i(128, 3, 49, 48, 47), i(124, 2, 50, 49),
            i(81, 2, 51, 44, 0), i(128, 2, 52, 51, 50),
        ] { words += instruction }
        if readWrite {
            words += i(98, 5, 53, 43, 24) + i(81, 2, 54, 53, 0) + i(128, 2, 55, 52, 54)
        }
        // Signed (-1, 0) coordinates must read zero and must leave the destination unchanged.
        words += i(98, 5, 57, 42, 26) + i(81, 2, 58, 57, 0) + i(128, 2, 59, readWrite ? 55 : 52, 58)
        words += i(80, 5, 56, 59, 23, 23, 23)
        words += i(99, failure == .writeReadOnly ? 42 : 43, 24, 56) + i(99, 43, 26, 56)
        return words + i(253) + i(56)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvStorageImagesPreserveFormatsAndAccess(_ format: ShaderOutputFormat, _ readWrite: Bool) throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.storageImageModule(readWrite: readWrite))
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.storageTextures.map(\.binding) == [IRResourceBinding(group: 0, binding: 0), .init(group: 0, binding: 1)])
    #expect(shader.storageTextures.map(\.format) == [.r32Uint, .r32Uint])
    #expect(shader.storageTextures.map(\.dimension) == [.d2, .d2])
    #expect(shader.storageTextures.map(\.access) == [.load, readWrite ? [.load, .store] : .store])
    #expect(shader.sampledTextures.isEmpty)
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(input)
    }
}

@Test(arguments: SPIRVFrontendTests.StorageImageFailure.allCases)
func spirvStorageImagesRejectInvalidDeclarationsAndOperations(_ failure: SPIRVFrontendTests.StorageImageFailure) throws {
    let words = SPIRVFrontendTests.storageImageModule(failure: failure)
    do {
        _ = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
        Issue.record("Accepted an invalid storage image")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        let diagnostic = try #require(diagnostics.first)
        guard case .binary(let span) = diagnostic.location else { Issue.record("Missing SPIR-V diagnostic location"); return }
        try span.validate(wordCount: words.count)
        try #require(words.indices.contains(span.startWord))
        #expect(words[span.startWord] & 0xffff == failure.offendingOpcode)
        if failure == .missingImageQueryCapability { #expect(diagnostic.code == .unsupportedCapability) }
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvStorageImagesReadWriteAndQueryOnMetal(_ readWrite: Bool) async throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.storageImageModule(readWrite: readWrite))
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint,
            usage: [.storageBinding, .copyDestination]))
        let destination = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint,
            usage: [.storageBinding, .copyDestination, .copySource]))
        let upload = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let region = TextureCopyRegion(width: 2, height: 1)
        let snapshots = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [11, 27, 101, 19]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: source, region: region),
            .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: destination, region: region),
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): source, .init(binding: 1): destination]),
            .copyTextureToBuffer(source: destination, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[101, readWrite ? 247 : 228]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
