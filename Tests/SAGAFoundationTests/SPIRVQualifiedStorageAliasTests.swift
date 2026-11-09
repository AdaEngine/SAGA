import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    enum QualifiedAliasAction {
        case mixed, readOnly, storeReadView, loadWriteView, loadWriteAggregate, storeReadViewAfterSelection
        case atomicReadView
    }

    // The first view uses signed words and the second unsigned words at the same offsets.
    static func qualifiedStorageAliasModule(members: Bool, action: QualifiedAliasAction = .mixed) -> [UInt32] {
        var words = pre + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        for block: UInt32 in [5, 6] {
            words += i(71, block, 2)
            for member: UInt32 in 0..<3 { words += i(72, block, member, 35, member * 4) }
        }
        for variable: UInt32 in [30, 31] { words += i(71, variable, 34, 0) + i(71, variable, 33, 0) + i(71, variable, 20) }
        if members {
            if action == .readOnly {
                for member: UInt32 in 0..<3 {
                    words += i(72, 5, member, 24) + i(72, 6, member, 24)
                }
            } else {
                words += i(72, 5, 0, 24) + i(72, 5, 1, 24)
                words += i(72, 6, 1, 25) + i(72, 6, 2, 25)
            }
        } else {
            words += i(71, 30, 24) + i(71, 31, action == .readOnly ? 24 : 25)
        }
        if action == .readOnly {
            words += i(71, 7, 2) + i(72, 7, 0, 35, 0) + i(71, 32, 34, 0) + i(71, 32, 33, 1)
        }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(20, 4), i(30, 5, 3, 3, 3), i(30, 6, 2, 2, 2),
            i(30, 7, 2), i(32, 8, 12, 5), i(32, 9, 12, 6), i(32, 10, 12, 7), i(32, 11, 12, 3),
            i(32, 12, 12, 2), i(33, 13, 1), i(43, 2, 20, 0), i(43, 2, 21, 1), i(43, 2, 22, 2),
            i(43, 3, 23, 5), i(43, 3, 24, 2), i(43, 2, 25, 0xabcdef01), i(43, 2, 26, 7), i(41, 4, 27),
            i(59, 8, 30, 12), i(59, 9, 31, 12),
        ] { words += instruction }
        if action == .readOnly { words += i(59, 10, 32, 12) }
        words += i(54, 1, 40, 0, 13) + i(248, 41)
        words += i(65, 11, 42, 30, 20) + i(65, 11, 43, 30, 21) + i(65, 12, 44, 31, 21) + i(65, 12, 45, 31, 22)
        switch action {
        case .mixed:
            words += i(61, 3, 46, 42) + i(128, 3, 47, 46, 23) + i(124, 2, 48, 47) + i(62, 44, 48)
            words += i(61, 3, 49, 43) + i(170, 4, 50, 49, 24) + i(169, 2, 51, 50, 25, 26) + i(62, 45, 51)
        case .readOnly:
            words += i(61, 3, 46, 42) + i(124, 2, 47, 46) + i(61, 2, 48, 44)
            words += i(128, 2, 49, 47, 48) + i(65, 12, 50, 32, 20) + i(62, 50, 49)
        case .storeReadView: words += i(62, 42, 23)
        case .loadWriteView: words += i(61, 2, 46, 44)
        case .loadWriteAggregate: words += i(61, 6, 46, 31)
        case .atomicReadView: words += i(234, 3, 46, 42, 21, 20, 23)
        case .storeReadViewAfterSelection:
            words += i(83, 11, 60, 42) + i(247, 63, 0) + i(250, 27, 61, 62)
            words += i(248, 61) + i(61, 3, 64, 60) + i(249, 63)
            words += i(248, 62) + i(61, 3, 65, 60) + i(249, 63)
            words += i(248, 63) + i(62, 60, 23)
        }
        return words + i(253) + i(56)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl], [false, true])
func spirvQualifiedStorageAliasMergesPhysicalAccess(_ format: ShaderOutputFormat, _ members: Bool) throws {
    let compiler = IndependentShaderCompiler()
    for readOnly in [false, true] {
        let words = SPIRVFrontendTests.qualifiedStorageAliasModule(members: members, action: readOnly ? .readOnly : .mixed)
        let input = ShaderInput.spirv(words)
        let shader = try compiler.compileCompute(input, to: format)
        let bindings = shader.storageBindings.sorted { $0.binding.binding < $1.binding.binding }
        #expect(bindings.map(\.binding.binding) == (readOnly ? [0, 1] : [0]))
        #expect(bindings.map(\.isReadOnly) == (readOnly ? [true, false] : [false]))
        if format == .wgsl {
            guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
            _ = try compiler.parseAndValidate(.wgsl(source))
            _ = try compiler.compileWebGPUCompute(input)
        }
    }
}

@Test(arguments: [false, true])
func spirvQualifiedStorageAliasRejectsOriginalViewViolations(_ members: Bool) throws {
    let compiler = IndependentShaderCompiler()
    for action: SPIRVFrontendTests.QualifiedAliasAction in [
        .storeReadView, .loadWriteView, .loadWriteAggregate, .storeReadViewAfterSelection, .atomicReadView,
    ] {
        let input = ShaderInput.spirv(SPIRVFrontendTests.qualifiedStorageAliasModule(members: members, action: action))
        do {
            _ = try compiler.compileCompute(input, to: .msl)
            Issue.record("Accepted a restricted storage alias access")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.first?.message.contains("NonWritable or NonReadable") == true)
        }
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvQualifiedStorageAliasPreservesReadAfterWriteOnMetal(_ members: Bool) async throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.qualifiedStorageAliasModule(members: members))
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let buffer = try await device.createBuffer(.init(size: 12, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: buffer, offset: 0, words: [0xfffffffd, 0, 0]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): buffer], workgroups: .init(x: 1)),
            .copy(source: buffer, destination: read, byteCount: 12), .read(buffer: read, byteCount: 12),
        ]).wait()
        #expect(snapshots == [[0xfffffffd, 2, 0xabcdef01]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
