import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing

#if canImport(Metal)
import SAGAMetal
#endif

extension SPIRVFrontendTests {
    static func signAdjustedIntegerModule(invalid: String? = nil) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 17, 0x6e69616d, 0), i(16, 17, 17, 1, 1, 1),
            i(71, 12, 6, 4), i(71, 13, 2), i(72, 13, 0, 35, 0),
            i(71, 16, 34, 0), i(71, 16, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(20, 4),
            i(23, 5, 2, 2), i(23, 6, 3, 2), i(23, 7, 4, 2),
            i(22, 8, 32), i(43, 8, 9, Float(1).bitPattern),
            i(33, 10, 1), i(43, 2, 11, 16), i(28, 12, 2, 11), i(30, 13, 12),
            i(32, 14, 12, 13), i(32, 15, 12, 2), i(59, 14, 16, 12),
        ] { words += instruction }
        for index: UInt32 in 0..<16 { words += i(43, 2, 20 + index, index) }
        words += i(54, 1, 17, 0, 10) + i(248, 18)

        var nextID: UInt32 = 40
        func value(_ opcode: UInt32, _ type: UInt32, _ operands: UInt32...) -> UInt32 {
            let id = nextID
            nextID += 1
            words += [UInt32(operands.count + 3) << 16 | opcode, type, id] + operands
            return id
        }
        func pointer(_ index: UInt32) -> UInt32 { value(65, 15, 16, 20, 20 + index) }
        func load(_ index: UInt32) -> UInt32 { value(61, 2, pointer(index)) }
        func booleanWord(_ predicate: UInt32) -> UInt32 { value(169, 2, predicate, 21, 20) }

        let high = load(0)
        let negativeBits = load(1)
        let negative = value(124, 3, negativeBits)
        let positive = load(2)
        let unsignedPair = value(80, 5, high, positive)
        let signedPair = value(80, 6, negative, negative)

        let signedNegate = value(126, 3, high)
        let unsignedNegate = value(126, 2, negative)
        let rhs = invalid == "shape" ? signedPair : invalid == "float" ? 9 : negative
        let add = value(invalid == "unsigned division" ? 134 : 128, 2, high, rhs)
        let subtract = value(130, 3, high, negative)
        let multiply = value(132, 2, high, negative)
        let unsignedLess = booleanWord(value(176, 4, positive, negative))
        let signedLess = booleanWord(value(177, 4, positive, negative))
        let equal = booleanWord(value(170, 4, negativeBits, negative))
        let notEqual = booleanWord(value(171, 4, high, negative))
        let pairAdd = value(128, 5, unsignedPair, signedPair)
        let pairLess = value(177, 7, unsignedPair, signedPair)
        let output = [
            value(124, 2, signedNegate), unsignedNegate, add, value(124, 2, subtract), multiply,
            unsignedLess, signedLess, equal, notEqual, value(81, 2, pairAdd, 0), value(81, 2, pairAdd, 1),
            booleanWord(value(81, 4, pairLess, 0)), booleanWord(value(81, 4, pairLess, 1)),
        ]
        for (index, result) in output.enumerated() {
            let destination = pointer(UInt32(index + 3))
            words += i(62, destination, result)
        }
        return words + i(253) + i(56)
    }

    @Test(arguments: ["shape", "float", "unsigned division"])
    func spirvSignAdjustedIntegerOperationsStillRejectOtherTypeChanges(_ invalid: String) throws {
        try fails(Self.signAdjustedIntegerModule(invalid: invalid), code: .invalidBinary)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvSignAdjustedIntegerOperationsReachTextBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.signAdjustedIntegerModule()),
                                             entryPoint: "main", to: format)
    guard case .text(let actual, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func spirvSignAdjustedIntegerOperationsPreserveBitsAndComparisonOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(
        .spirv(SPIRVFrontendTests.signAdjustedIntegerModule()), entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input: [UInt32] = [0x80000001, 0xfffffff7, 7]
        let expected: [UInt32] = input + [
            0x7fffffff, 9, 0x7ffffff8, 0x8000000a, 0x7ffffff7, 1, 0, 1, 1, 0x7ffffff8, 0xfffffffe, 1, 0,
        ]
        let bytes: UInt64 = 64
        let data = try await device.createBuffer(.init(size: bytes, usage: [.storage, .copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: bytes, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .write(buffer: data, offset: 0, words: input),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data], workgroups: .init(x: 1)),
            .copy(source: data, destination: readback, byteCount: bytes), .read(buffer: readback, byteCount: bytes),
        ]).wait()
        #expect(result == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
