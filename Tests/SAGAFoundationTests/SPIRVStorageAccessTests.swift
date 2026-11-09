import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    enum StorageAccessDecoration {
        case variable(UInt32)
        case allMembers(UInt32)
        case nestedMember(UInt32)
    }

    enum StorageAccessAction {
        case readProtected, readSibling, readAggregate, writeProtected, writeSibling, writeAggregate, atomicProtected
    }

    static func storageAccessModule(
        _ decoration: StorageAccessDecoration, action: StorageAccessAction = .readProtected,
        legacy: Bool = false, uniformSource: Bool = false
    ) -> [UInt32] {
        let sourceSpace: UInt32 = legacy || uniformSource ? 2 : 12
        let outputSpace: UInt32 = legacy ? 2 : 12
        let block: UInt32 = legacy ? 3 : 2
        var words = pre + i(15, 5, 20, 0x6e69616d, 0) + i(16, 20, 17, 1, 1, 1)
        for instruction in [
            i(71, 4, block), i(71, 5, block), i(72, 3, 0, 35, 0), i(72, 3, 1, 35, 4),
            i(72, 4, 0, 35, 0), i(72, 4, 1, 35, 8), i(72, 5, 0, 35, 0),
            i(71, 15, 34, 0), i(71, 15, 33, 0), i(71, 16, 34, 0), i(71, 16, 33, 1),
        ] { words += instruction }
        switch decoration {
        case .variable(let flag): words += i(71, 15, flag)
        case .allMembers(let flag): words += i(72, 4, 0, flag) + i(72, 4, 1, flag)
        case .nestedMember(let flag): words += i(72, 3, 0, flag)
        }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(30, 3, 2, 2), i(30, 4, 3, 2), i(30, 5, 2),
            i(32, 6, sourceSpace, 4), i(32, 7, sourceSpace, 3), i(32, 8, sourceSpace, 2),
            i(32, 9, outputSpace, 5), i(32, 17, outputSpace, 2), i(33, 10, 1),
            i(43, 2, 12, 0), i(43, 2, 13, 1), i(43, 2, 14, 7),
            i(59, 6, 15, sourceSpace), i(59, 9, 16, outputSpace), i(54, 1, 20, 0, 10), i(248, 21),
            i(65, 7, 22, 15, 12), i(65, 8, 23, 22, 12), i(83, 8, 24, 23), i(65, 8, 25, 22, 13),
        ] { words += instruction }
        switch action {
        case .readProtected, .readSibling:
            words += i(61, 2, 26, action == .readProtected ? 24 : 25)
            words += i(65, 17, 27, 16, 12) + i(62, 27, 26)
        case .readAggregate: words += i(61, 3, 26, 22)
        case .writeProtected: words += i(62, 24, 14)
        case .writeSibling: words += i(62, 25, 14)
        case .writeAggregate: words += i(80, 3, 26, 14, 14) + i(62, 22, 26)
        case .atomicProtected: words += i(234, 2, 26, 24, 13, 12, 14)
        }
        return words + i(253) + i(56)
    }

    @Test func spirvStorageAccessRejectsWritesToProtectedMembersAndAggregates() throws {
        for action: StorageAccessAction in [.writeProtected, .writeAggregate, .atomicProtected] {
            try fails(Self.storageAccessModule(.nestedMember(24), action: action))
        }
        try fails(Self.storageAccessModule(.variable(24), action: .writeSibling))
        try fails(Self.storageAccessModule(.allMembers(24), action: .writeSibling, legacy: true))
        _ = try parse(Self.storageAccessModule(.nestedMember(24), action: .writeSibling))
        try fails(Self.storageAccessModule(.variable(24), uniformSource: true))
    }

    @Test func spirvStorageAccessRejectsReadsOfNonReadableMembersAndAggregates() throws {
        for action: StorageAccessAction in [.readProtected, .readAggregate, .atomicProtected] {
            try fails(Self.storageAccessModule(.nestedMember(25), action: action))
        }
        try fails(Self.storageAccessModule(.variable(25), action: .readSibling))
        _ = try parse(Self.storageAccessModule(.nestedMember(25), action: .readSibling))
        _ = try parse(Self.storageAccessModule(.variable(25), action: .writeSibling, legacy: true))
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl], [false, true])
func spirvStorageAccessReflectsWholeReadonlyBuffers(_ format: ShaderOutputFormat, _ legacy: Bool) throws {
    let compiler = IndependentShaderCompiler()
    for decoration: SPIRVFrontendTests.StorageAccessDecoration in [.variable(24), .allMembers(24)] {
        let fixture = ShaderInput.spirv(SPIRVFrontendTests.storageAccessModule(decoration, legacy: legacy))
        let shader = try compiler.compileCompute(fixture, to: format)
        #expect(shader.storageBindings.map(\.isReadOnly) == [true, false])
        guard case .text(let target, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(target == format)
        if format == .wgsl {
            #expect(source.contains("var<storage, read>"))
            _ = try compiler.parseAndValidate(.wgsl(source))
            _ = try compiler.compileWebGPUCompute(fixture)
        }
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvStorageAccessReadsProtectedFieldsOnMetal(_ legacy: Bool) async throws {
    let fixture = ShaderInput.spirv(SPIRVFrontendTests.storageAccessModule(.allMembers(24), legacy: legacy))
    let shader = try IndependentShaderCompiler().compileCompute(fixture, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 12, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [17, 20, 31]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 4), .read(buffer: read, byteCount: 4),
        ]).wait()
        #expect(result == [[17]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
