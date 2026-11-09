import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func runtimeStorageLayoutModule(
        stride: UInt32 = 4, tailOffset: UInt32 = 32, tailIsFinal: Bool = true, nestedTail: Bool = false,
        lengthMemberIndex: UInt32 = 1
    ) -> [UInt32] {
        var words = pre + i(15, 5, 30, 0x6e69616d, 0) + i(16, 30, 17, 1, 1, 1)
        words += i(71, 5, 6, stride) + i(71, 6, 2) + i(71, 7, 2)
        words += i(72, 6, 0, 35, 0) + i(72, 6, 1, 35, tailOffset) + i(72, 7, 0, 35, 0) + i(72, 7, 1, 35, 4)
        if !tailIsFinal { words += i(72, 6, 2, 35, 48) }
        if nestedTail { words += i(72, 9, 0, 35, 0) }
        words += i(71, 20, 34, 0) + i(71, 20, 33, 0) + i(71, 21, 34, 0) + i(71, 21, 33, 1)
        words += i(19, 1) + i(21, 2, 32, 0) + i(23, 4, 2, 4) + i(29, 5, 2)
        if nestedTail { words += i(30, 9, 5) }
        let tailType: UInt32 = nestedTail ? 9 : 5
        words += tailIsFinal ? i(30, 6, 4, tailType) : i(30, 6, 4, tailType, 2)
        for instruction in [
            i(30, 7, 2, 2), i(32, 8, 12, 6), i(32, 10, 12, 7), i(32, 11, 12, 2), i(33, 12, 1),
            i(43, 2, 22, 0), i(43, 2, 23, 1), i(59, 8, 20, 12), i(59, 10, 21, 12),
            i(54, 1, 30, 0, 12), i(248, 31),
            i(65, 11, 33, 20, 22, 22), i(61, 2, 34, 33), i(65, 11, 35, 20, 22, 23), i(61, 2, 36, 35),
            i(65, 11, 37, 20, 23, 36), i(61, 2, 38, 37), i(128, 2, 39, 34, 38),
            i(65, 11, 40, 21, 22), i(62, 40, 39), i(68, 2, 41, 20, lengthMemberIndex),
            i(65, 11, 42, 21, 23), i(62, 42, 41), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvRuntimeStorageLayoutRejectsNonFinalAndNestedArrays() throws {
        try fails(Self.runtimeStorageLayoutModule(tailIsFinal: false))
        try fails(Self.runtimeStorageLayoutModule(nestedTail: true))
    }

    @Test func spirvRuntimeStorageLayoutRejectsInvalidStrideAndSpan() throws {
        for stride: UInt32 in [2, 6] { try fails(Self.runtimeStorageLayoutModule(stride: stride)) }
        for offset: UInt32 in [12, 33, 0x7ffffffc, 0xfffffffc] {
            try fails(Self.runtimeStorageLayoutModule(tailOffset: offset))
        }
    }

    @Test func spirvArrayLengthRejectsHeaderAndOutOfRangeMember() throws {
        try fails(Self.runtimeStorageLayoutModule(lengthMemberIndex: 0))
        try fails(Self.runtimeStorageLayoutModule(lengthMemberIndex: 2))
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvRuntimeStorageLayoutPreservesHeaderPaddingAndTailReflection(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.runtimeStorageLayoutModule())
    let module = try compiler.parseAndValidate(input).module
    let buffer = try #require(module.globalVariables.values.first { $0.binding == .init(group: 0, binding: 0) })
    guard case .structure(let members, let span) = try module.types.value(for: buffer.type).kind else {
        throw ShaderFailure.incompatibleArtifact
    }
    #expect(members.map(\.offset) == [0, 32])
    #expect(span == 48)
    let tail = try #require(members.last)
    guard case .array(_, .dynamic, let stride) = try module.types.value(for: tail.type).kind else {
        throw ShaderFailure.incompatibleArtifact
    }
    #expect(stride == 4)

    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [48, 8])
    #expect(shader.storageBindings.map(\.runtimeArrayOffset) == [32, 0])
    #expect(shader.storageBindings.map(\.runtimeArrayStride) == [4, nil])
    if format == .wgsl, case .text(_, let source) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.compileWebGPUCompute(input)
    }
}

#if canImport(Metal)
@Test func spirvRuntimeStorageLayoutUsesTailAndBoundLengthOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(
        .spirv(SPIRVFrontendTests.runtimeStorageLayoutModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [7, 2, 0, 0, 100, 101, 102, 103, 4, 10, 12, 33, 99, 0, 0, 0]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
            .write(buffer: input, offset: 4, words: [4]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0): .init(buffer: input, offset: 0, size: 48),
                .init(binding: 1): .init(buffer: output, offset: 0, size: 8),
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[19, 8], [7, 4]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
