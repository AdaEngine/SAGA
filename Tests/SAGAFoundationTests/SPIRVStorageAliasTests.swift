import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    enum StorageAliasScalar: CaseIterable, Sendable {
        case signed16, signed32, float32

        var narrow: Bool { self == .signed16 }
        var stride: UInt32 { narrow ? 2 : 4 }

        var primaryType: UInt32 { self == .signed32 ? 3 : 5 }
        var unsignedType: UInt32 { narrow ? 6 : 2 }
        var initialBits: UInt32 {
            switch self {
            case .signed16: 0xfffe
            case .signed32: 0xfffffffe
            case .float32: Float(-2.25).bitPattern
            }
        }
        var primaryWrite: UInt32 {
            self == .float32 ? Float(-7.5).bitPattern : UInt32(bitPattern: -7)
        }
        var unsignedWrite: UInt32 {
            switch self {
            case .signed16: 0xabcd
            case .signed32: 0xabcdef01
            case .float32: Float(1.5).bitPattern
            }
        }
        var input: [UInt32] {
            narrow ? [initialBits, 0, 0xbeef0000] : [initialBits, 0, 0, 0, 0, 0xdeadbeef]
        }
        var output: [UInt32] {
            narrow ? [0xfff9fffe, 0xfff9abcd, 0xbeefabcd] :
                [initialBits, primaryWrite, unsignedWrite, primaryWrite, unsignedWrite, 0xdeadbeef]
        }
    }

    // Two independent runtime-array block types with the same descriptor binding.
    // The shader performs an actual signed/unsigned test and reads each store through the other view.
    static func storageAliasModule(
        _ scalar: StorageAliasScalar, legacy: Bool = false, mismatchedStride: Bool = false,
        interfaceList: [UInt32]? = nil, disjointEntryPoints: Bool = false
    ) -> [UInt32] {
        let primary = scalar.primaryType
        let unsigned = scalar.unsignedType
        let storage: UInt32 = legacy ? 2 : 12
        let block: UInt32 = legacy ? 3 : 2
        var words = head + i(17, 1)
        if scalar.narrow { words += i(17, 22) + i(17, 4433) }
        words += i(14, 0, 1)
        if let interfaceList {
            words[1] = 0x00010400
            words += [UInt32(5 + interfaceList.count) << 16 | 15, 5, 33, 0x6e69616d, 0] + interfaceList
        } else {
            words += i(15, 5, 33, 0x6e69616d, 0)
        }
        if disjointEntryPoints { words += i(15, 5, 60, 0x00746c61) }
        words += i(16, 33, 17, 1, 1, 1)
        if disjointEntryPoints { words += i(16, 60, 17, 1, 1, 1) }
        words += i(71, 7, 6, scalar.stride) + i(71, 8, 6, mismatchedStride ? scalar.stride * 2 : scalar.stride)
        words += i(71, 9, block) + i(71, 10, block) + i(72, 9, 0, 35, 0) + i(72, 10, 0, 35, 0)
        for variable: UInt32 in [31, 32] { words += i(71, variable, 34, 0) + i(71, variable, 33, 0) + i(71, variable, 20) }
        words += i(19, 1) + i(21, 2, 32, 0) + i(21, 3, 32, 1) + i(20, 4)
        if scalar.narrow { words += i(21, 5, 16, 1) + i(21, 6, 16, 0) }
        if scalar == .float32 { words += i(22, 5, 32) }
        for instruction in [
            i(29, 7, primary), i(29, 8, unsigned), i(30, 9, 7), i(30, 10, 8),
            i(32, 11, storage, 9), i(32, 12, storage, 10), i(32, 13, storage, primary), i(32, 14, storage, unsigned), i(33, 15, 1),
        ] { words += instruction }
        for index: UInt32 in 0..<5 { words += i(43, 2, 20 + index, index) }
        words += i(43, scalar == .float32 ? primary : 3, 25, 0) + i(43, 2, 26, scalar.initialBits)
        words += i(43, primary, 27, scalar.primaryWrite) + i(43, primary, 28, scalar == .float32 ? Float(111).bitPattern : 111)
        words += i(43, unsigned, 29, scalar.unsignedWrite) + i(43, unsigned, 30, 55)
        words += i(59, 11, 31, storage) + i(59, 12, 32, storage) + i(54, 1, 33, 0, 15) + i(248, 34)
        if disjointEntryPoints {
            words += i(65, 13, 35, 31, 20, 20) + i(62, 35, 27) + i(253) + i(56)
            words += i(54, 1, 60, 0, 15) + i(248, 61)
            words += i(65, 14, 62, 32, 20, 20) + i(62, 62, 29)
            return words + i(253) + i(56)
        }
        if interfaceList != nil {
            words += i(65, 14, 37, 32, 20, 20) + i(61, unsigned, 38, 37)
            words += i(65, 14, 48, 32, 20, 21) + i(62, 48, 38)
            return words + i(253) + i(56)
        }
        words += i(65, 13, 35, 31, 20, 20) + i(61, primary, 36, 35)
        words += i(65, 14, 37, 32, 20, 20) + i(61, unsigned, 38, 37)
        if scalar.narrow {
            words += i(114, 3, 39, 36) + i(113, 2, 40, 38)
        }
        words += i(scalar == .float32 ? 184 : 177, 4, 43, scalar.narrow ? 39 : 36, 25)
        words += i(170, 4, 44, scalar.narrow ? 40 : 38, 26)
        words += i(169, primary, 45, 43, 27, 28) + i(169, unsigned, 46, 44, 29, 30)
        words += i(65, 13, 47, 31, 20, 21) + i(62, 47, 45)
        words += i(65, 14, 48, 32, 20, 22) + i(62, 48, 46)
        words += i(65, 14, 49, 32, 20, 21) + i(61, unsigned, 50, 49)
        words += i(65, 14, 51, 32, 20, 23) + i(62, 51, 50)
        words += i(65, 13, 52, 31, 20, 22) + i(61, primary, 53, 52)
        words += i(65, 13, 54, 31, 20, 24) + i(62, 54, 53)
        return words + i(253) + i(56)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl], SPIRVFrontendTests.StorageAliasScalar.allCases)
func spirvStorageAliasCoalescesViewsAcrossTextTargets(_ format: ShaderOutputFormat, _ scalar: SPIRVFrontendTests.StorageAliasScalar) throws {
    let compiler = IndependentShaderCompiler()
    for legacy in [false, true] {
        let fixture = ShaderInput.spirv(SPIRVFrontendTests.storageAliasModule(scalar, legacy: legacy))
        let shader = try compiler.compileCompute(fixture, to: format)
        #expect(shader.requiresShaderInt16 == scalar.narrow)
        let binding = try #require(shader.storageBindings.first)
        #expect(shader.storageBindings.count == 1)
        #expect(binding.binding == .init(group: 0, binding: 0))
        #expect(binding.runtimeArrayStride == scalar.stride)
        #expect(!binding.isReadOnly)
        guard case .text(let target, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(target == format)
        if format == .wgsl {
            _ = try compiler.parseAndValidate(.wgsl(source), capabilities: scalar.narrow ? [.shaderInt16] : [])
            if scalar.narrow {
                #expect(throws: ShaderFailure.self) { _ = try compiler.compileWebGPUCompute(fixture) }
            } else {
                let portable = try compiler.compileWebGPUCompute(fixture)
                #expect(portable.storageBindings.count == 1)
            }
        }
    }
}

@Test func spirvStorageAliasRejectsIncompatibleLayouts() throws {
    let compiler = IndependentShaderCompiler()
    let words = SPIRVFrontendTests.storageAliasModule(.signed16, mismatchedStride: true)
    do {
        _ = try compiler.compileCompute(.spirv(words), to: .msl)
        Issue.record("Accepted incompatible storage alias layouts")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .invalidBinding)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl])
func spirvStorageAliasAllowsDifferentLayoutsInDisjointEntryPoints(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fixture = ShaderInput.spirv(SPIRVFrontendTests.storageAliasModule(.signed32, mismatchedStride: true,
                                                                       disjointEntryPoints: true))
    let first = try compiler.compileCompute(fixture, entryPoint: "main", to: format)
    let second = try compiler.compileCompute(fixture, entryPoint: "alt", to: format)
    #expect(first.storageBindings.map(\.binding) == [.init(group: 0, binding: 0)])
    #expect(second.storageBindings.map(\.binding) == [.init(group: 0, binding: 0)])
    #expect(first.storageBindings.map(\.runtimeArrayStride) == [4])
    #expect(second.storageBindings.map(\.runtimeArrayStride) == [8])
}

@Test func spirvStorageAliasChecksOriginalVariablesIn14EntryInterface() throws {
    let compiler = IndependentShaderCompiler()
    let listed = SPIRVFrontendTests.storageAliasModule(.signed32, interfaceList: [32])
    let shader = try compiler.compileCompute(.spirv(listed), to: .wgsl)
    #expect(shader.storageBindings.count == 1)
    let unlisted = SPIRVFrontendTests.storageAliasModule(.signed32, interfaceList: [31])
    #expect(throws: ShaderFailure.self) { _ = try compiler.compileCompute(.spirv(unlisted), to: .wgsl) }
}

#if canImport(Metal)
@Test(arguments: SPIRVFrontendTests.StorageAliasScalar.allCases)
func spirvStorageAliasPreservesReinterpretationAndCrossViewStoresOnMetal(_ scalar: SPIRVFrontendTests.StorageAliasScalar) async throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.storageAliasModule(scalar, legacy: scalar.narrow))
    let shader = try compiler.compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: scalar.narrow ? [.shaderInt16] : [])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let byteCount = UInt64(scalar.input.count * MemoryLayout<UInt32>.size)
        let buffer = try await device.createBuffer(.init(size: byteCount, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: byteCount, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: buffer, offset: 0, words: scalar.input),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): buffer], workgroups: .init(x: 1)),
            .copy(source: buffer, destination: read, byteCount: byteCount), .read(buffer: read, byteCount: byteCount),
        ]).wait()
        #expect(snapshots == [scalar.output])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
