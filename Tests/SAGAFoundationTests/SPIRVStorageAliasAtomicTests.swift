import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func atomicAliasModule(canonicalSigned: Bool, legacy: Bool) -> [UInt32] {
        let canonical: UInt32 = canonicalSigned ? 3 : 2
        let alias: UInt32 = canonicalSigned ? 2 : 3
        let unsignedPointer: UInt32 = canonicalSigned ? 13 : 12
        let storageClass: UInt32 = legacy ? 2 : 12
        let block: UInt32 = legacy ? 3 : 2
        let minMax: UInt32 = canonicalSigned ? 236 : 239
        var words = head + i(17, 1) + i(14, 0, 1) + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        if legacy { words[1] = 0x00010000 }
        words += i(71, 4, 6, 4) + i(71, 5, 6, 4)
        words += i(71, 6, block) + i(72, 6, 0, 35, 0) + i(71, 7, block) + i(72, 7, 0, 35, 0) + i(71, 8, block)
        for member: UInt32 in 0..<8 { words += i(72, 8, member, 35, member * 4) }
        for id: UInt32 in [15, 16] { words += i(71, id, 34, 0) + i(71, id, 33, 0) + i(71, id, 20) }
        words += i(71, 17, 34, 0) + i(71, 17, 33, 1)
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(29, 4, canonical), i(29, 5, alias),
            i(30, 6, 4), i(30, 7, 5), i(30, 8, 2, 2, 2, 2, 2, 2, 2, 2),
            i(32, 9, storageClass, 6), i(32, 10, storageClass, 7), i(32, 11, storageClass, 8),
            i(32, 12, storageClass, canonical), i(32, 13, storageClass, alias), i(33, 14, 1),
        ] { words += instruction }
        for value: UInt32 in 0..<8 { words += i(43, 2, 20 + value, value) }
        for (index, bits) in [UInt32(0x40000000), 0xc0000001, 0xf0000001, .max, 0x10000002, 0xfffffff0, 0xfffffffe].enumerated() {
            words += i(43, alias, 28 + UInt32(index), bits)
        }
        for instruction in [
            i(59, 9, 15, storageClass), i(59, 10, 16, storageClass), i(59, 11, 17, storageClass),
            i(54, 1, 40, 0, 14), i(248, 41),
            i(65, 13, 42, 16, 20, 20), i(65, 13, 43, 16, 20, 21),
            i(65, 12, 44, 15, 20, 20), i(65, 12, 45, 15, 20, 21),
            i(227, alias, 50, 42, 21, 20), i(241, alias, 51, 42, 21, 20, 28),
            i(230, alias, 52, 42, 21, 20, 20, 30, 29), i(230, alias, 53, 42, 21, 20, 20, 31, 29),
            i(234, alias, 54, 42, 21, 20, 32), i(228, 43, 21, 20, 33), i(227, canonical, 55, 45, 21, 20),
            i(minMax, alias, 56, 42, 21, 20, 34), i(227, canonical, 57, 44, 21, 20),
        ] { words += instruction }
        for offset: UInt32 in 0..<8 {
            let raw = 50 + offset
            let rawType = offset == 5 || offset == 7 ? canonical : alias
            let bits = rawType == 2 ? raw : 60 + offset
            if rawType != 2 { words += i(124, 2, bits, raw) }
            words += i(65, unsignedPointer, 70 + offset, 17, 20 + offset) + i(62, 70 + offset, bits)
        }
        return words + i(253) + i(56)
    }

    // The second buffer supplies an index and captures the old min/max values and the final value.
    static func atomicAliasOppositeOrderingModule(canonicalSigned: Bool) -> [UInt32] {
        let canonical: UInt32 = canonicalSigned ? 3 : 2
        let alias: UInt32 = canonicalSigned ? 2 : 3
        let min: UInt32 = canonicalSigned ? 237 : 236
        let max: UInt32 = canonicalSigned ? 239 : 238
        let minOperand: UInt32 = canonicalSigned ? 0x10000002 : 0x80000002
        let maxOperand: UInt32 = canonicalSigned ? 0x80000003 : 0x10000003
        var words = pre + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        words += i(71, 4, 6, 4) + i(71, 5, 6, 4)
        words += i(71, 6, 2) + i(72, 6, 0, 35, 0) + i(71, 7, 2) + i(72, 7, 0, 35, 0) + i(71, 8, 2)
        for member: UInt32 in 0..<4 { words += i(72, 8, member, 35, member * 4) }
        for id: UInt32 in [15, 16] { words += i(71, id, 34, 0) + i(71, id, 33, 0) + i(71, id, 20) }
        words += i(71, 17, 34, 0) + i(71, 17, 33, 1)
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(29, 4, canonical), i(29, 5, alias),
            i(30, 6, 4), i(30, 7, 5), i(30, 8, 2, 2, 2, 2),
            i(32, 9, 12, 6), i(32, 10, 12, 7), i(32, 11, 12, 8),
            i(32, 12, 12, canonical), i(32, 13, 12, alias), i(33, 14, 1),
        ] { words += instruction }
        for value: UInt32 in 0..<4 { words += i(43, 2, 20 + value, value) }
        words += i(43, alias, 24, minOperand) + i(43, alias, 25, maxOperand)
        let unsignedPointer: UInt32 = canonicalSigned ? 13 : 12
        for instruction in [
            i(59, 9, 15, 12), i(59, 10, 16, 12), i(59, 11, 17, 12),
            i(54, 1, 40, 0, 14), i(248, 41), i(65, unsignedPointer, 42, 17, 20), i(61, 2, 43, 42),
            i(65, 13, 44, 16, 20, 43), i(65, 12, 45, 15, 20, 43),
            i(min, alias, 46, 44, 21, 20, 24), i(max, alias, 47, 44, 21, 20, 25), i(227, canonical, 48, 45, 21, 20),
        ] { words += instruction }
        for offset: UInt32 in 1...3 {
            let raw = 45 + offset
            let rawType = offset == 3 ? canonical : alias
            let bits = rawType == 2 ? raw : 60 + offset
            if rawType != 2 { words += i(124, 2, bits, raw) }
            words += i(65, unsignedPointer, 70 + offset, 17, 20 + offset) + i(62, 70 + offset, bits)
        }
        return words + i(253) + i(56)
    }
}

@Test(arguments: [true, false])
func spirvStorageAliasAtomicMinMaxPreservesOppositeOrderingInTextBackends(_ canonicalSigned: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.atomicAliasOppositeOrderingModule(canonicalSigned: canonicalSigned))
    for format: ShaderOutputFormat in [.msl, .wgsl, .glsl, .hlsl] {
        let shader = try compiler.compileCompute(input, to: format)
        #expect(shader.storageBindings.map(\.binding) == [IRResourceBinding(group: 0, binding: 0), .init(group: 0, binding: 1)])
        if format == .wgsl {
            guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
            #expect(text.contains(canonicalSigned ? "atomic<i32>" : "atomic<u32>"))
            _ = try compiler.parseAndValidate(.wgsl(text))
            _ = try compiler.compileWebGPUCompute(input)
        }
    }
}

@Test(arguments: [true, false])
func spirvStorageAliasAtomicUsesCanonicalTypeInTextBackends(_ canonicalSigned: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.spirv(SPIRVFrontendTests.atomicAliasModule(canonicalSigned: canonicalSigned, legacy: false))
    for format: ShaderOutputFormat in [.msl, .wgsl, .glsl, .hlsl] {
        let artifact = try compiler.compileCompute(source, to: format)
        #expect(artifact.storageBindings.map(\.binding) == [IRResourceBinding(group: 0, binding: 0), .init(group: 0, binding: 1)])
        guard case .text(let target, let text) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(target == format)
        if format == .wgsl {
            #expect(text.contains(canonicalSigned ? "atomic<i32>" : "atomic<u32>"))
            #expect(text.contains("atomicCompareExchangeWeak("))
            _ = try compiler.parseAndValidate(.wgsl(text))
            _ = try compiler.compileWebGPUCompute(source)
        }
    }
}

#if canImport(Metal)
@Test(arguments: [true, false], [UInt32(1), 3, .max])
func spirvStorageAliasAtomicMinMaxPreservesOppositeOrderingAndBoundsOnMetal(_ canonicalSigned: Bool, _ index: UInt32) async throws {
    let words = SPIRVFrontendTests.atomicAliasOppositeOrderingModule(canonicalSigned: canonicalSigned)
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
    let initial: UInt32 = canonicalSigned ? 0x80000001 : 0x40000000
    let minimum: UInt32 = canonicalSigned ? 0x10000002 : 0x80000002
    let maximum: UInt32 = canonicalSigned ? 0x80000003 : 0x10000003
    let inBounds = index == 1
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let storage = try await device.createBuffer(.init(size: 12, usage: [.storage, .copySource, .copyDestination]))
        let results = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: storage, offset: 0, words: [0xaced1234, initial, 0xf00d5555]),
            .write(buffer: results, offset: 0, words: [index, .max, .max, .max]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): storage, .init(binding: 1): results],
                             workgroups: .init(x: 1)),
            .copy(source: results, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
            .copy(source: storage, destination: read, byteCount: 12), .read(buffer: read, byteCount: 12),
        ]).wait()
        #expect(snapshots == [
            inBounds ? [index, initial, minimum, maximum] : [index, 0, 0, 0],
            [0xaced1234, inBounds ? maximum : initial, 0xf00d5555],
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(arguments: [true, false], [true, false])
func spirvStorageAliasAtomicReturnsOriginalBitsAcrossViewsOnMetal(_ canonicalSigned: Bool, _ legacy: Bool) async throws {
    let words = SPIRVFrontendTests.atomicAliasModule(canonicalSigned: canonicalSigned, legacy: legacy)
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let storage = try await device.createBuffer(.init(size: 12, usage: [.storage, .copySource, .copyDestination]))
        let results = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: storage, offset: 0, words: [0x80000001, 0xabcdef01, 0xface1234]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): storage, .init(binding: 1): results],
                             workgroups: .init(x: 1)),
            .copy(source: results, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
            .copy(source: storage, destination: read, byteCount: 12), .read(buffer: read, byteCount: 12),
        ]).wait()
        #expect(snapshots == [
            [0x80000001, 0x80000001, 0xc0000001, 0xf0000001, 0xf0000001, 0xfffffff0, 3, 0xfffffffe],
            [0xfffffffe, 0xfffffff0, 0xface1234],
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
