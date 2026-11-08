import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func floatAtomicModule(
        capability: Bool = true, extensionDeclared: Bool = true, semantics: UInt32 = 0, addOpcode: UInt32 = 6035
    ) -> [UInt32] {
        var words = head + i(17, 1)
        if capability { words += i(17, 6033) }
        if extensionDeclared {
            // SPV_EXT_shader_atomic_float_add with a NUL terminator.
            words += i(10, 0x5f565053, 0x5f545845, 0x64616873, 0x615f7265, 0x696d6f74, 0x6c665f63, 0x5f74616f, 0x646461)
        }
        for instruction in [
            i(14, 0, 1), i(15, 5, 20, 0x6e69616d, 0, 17), i(16, 20, 17, 1, 1, 1),
            i(4, 0x455f4c47, 0x735f5458, 0x65646168, 0x74615f72, 0x63696d6f, 0x6f6c665f, 0x7461),
            i(71, 17, 11, 28), i(71, 5, 6, 4), i(71, 6, 2), i(72, 6, 0, 35, 0), i(72, 6, 1, 35, 4),
            i(71, 7, 2), i(72, 7, 0, 35, 0), i(72, 7, 1, 35, 4), i(72, 7, 2, 35, 8),
            i(72, 7, 3, 35, 12), i(72, 7, 4, 35, 16),
            i(71, 15, 34, 0), i(71, 15, 33, 0), i(71, 16, 34, 0), i(71, 16, 33, 1),
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 2, 3),
            i(43, 2, 30, 0), i(43, 2, 31, 1), i(43, 2, 32, 2), i(43, 2, 33, 3), i(43, 2, 34, 4),
            i(43, 3, 35, 0x80000000), i(43, 3, 36, 0x3f000000), i(43, 3, 37, 0x40e00000),
            i(43, 3, 38, 0x40100000), i(43, 2, 39, semantics),
            i(28, 5, 3, 31), i(30, 6, 5, 3), i(30, 7, 2, 2, 2, 2, 2),
            i(32, 8, 12, 6), i(32, 9, 12, 7), i(32, 10, 12, 3), i(32, 11, 12, 2),
            i(32, 12, 1, 4), i(32, 13, 1, 2), i(33, 14, 1),
            i(59, 8, 15, 12), i(59, 9, 16, 12), i(59, 12, 17, 1), i(54, 1, 20, 0, 14), i(248, 21),
            i(65, 10, 50, 15, 30, 30), i(228, 50, 31, 30, 35),
            i(addOpcode, 3, 51, 50, 31, 39, 36), i(229, 3, 52, 50, 31, 30, 37),
            i(65, 13, 53, 17, 30), i(61, 2, 54, 53), i(128, 2, 55, 54, 31),
            i(65, 10, 56, 15, 30, 55), i(addOpcode, 3, 57, 56, 31, 39, 36),
            i(227, 3, 58, 50, 31, 30), i(65, 10, 59, 15, 31), i(62, 59, 38), i(61, 3, 60, 59),
            i(124, 2, 61, 51), i(124, 2, 62, 52), i(124, 2, 63, 58), i(124, 2, 64, 57), i(124, 2, 65, 60),
            i(65, 11, 70, 16, 30), i(62, 70, 61), i(65, 11, 71, 16, 31), i(62, 71, 62),
            i(65, 11, 72, 16, 32), i(62, 72, 63), i(65, 11, 73, 16, 33), i(62, 73, 64),
            i(65, 11, 74, 16, 34), i(62, 74, 65), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func spirvFloatAtomicAddRequiresExtensionAndCapability() throws {
        try fails(Self.floatAtomicModule(capability: false), code: .unsupportedCapability)
        try fails(Self.floatAtomicModule(extensionDeclared: false), code: .unsupportedCapability)
        try fails(Self.floatAtomicModule(semantics: 0x48), code: .unsupportedConstruct)
        try fails(Self.floatAtomicModule(addOpcode: 234), code: .unsupportedConstruct)
    }

    @Test func spirvFloatAtomicUpgradePreservesOrdinaryFloatMember() throws {
        let module = try IndependentShaderCompiler().parseAndValidate(
            .spirv(Self.floatAtomicModule()), capabilities: [.shaderFloat32Atomic]).module
        let counter = try #require(module.globalVariables.values.first)
        guard case .structure(let fields, _) = try module.types.value(for: counter.type).kind,
              case .array(let element, .constant(1), 4) = try module.types.value(for: fields[0].type).kind else {
            throw ShaderFailure.invalidHandle
        }
        #expect(try module.types.value(for: element).kind == .atomic(.f32))
        #expect(try module.types.value(for: fields[1].type).kind == .scalar(.f32))
    }

    @Test func spirvFloatAtomicCoreInstructionsCompileWithoutAddExtension() throws {
        let words = Self.floatAtomicModule(capability: false, extensionDeclared: false, addOpcode: 229)
        _ = try IndependentShaderCompiler().compileCompute(.spirv(words), to: .wgsl)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvFloatAtomicOperationsTranslate(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.floatAtomicModule()), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [8, 20])
    guard case .text(let actual, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code)) }
}

#if canImport(Metal)
@Test func spirvFloatAtomicOperationsPreservePreviousBitsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.spirv(SPIRVFrontendTests.floatAtomicModule()), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let counter = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let output = try await device.createBuffer(.init(size: 20, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 20, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): counter, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 20), .read(buffer: readback, byteCount: 20),
            .copy(source: counter, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[0x80000000, 0x3f000000, 0x40e00000, 0, 0x40100000], [0x40e00000, 0x40100000]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
