import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func primitiveIndexModule(
        capability: UInt32? = 2, primitiveType: UInt32 = 2, model: UInt32 = 4, input: Bool = true, flat: Bool = true
    ) -> [UInt32] {
        var words = head + i(17, 1)
        if let capability { words += i(17, capability) }
        words += i(14, 0, 1) + i(15, model, 20, 0x6e69616d, 0, 12, 13)
        if model == 4 { words += i(16, 20, 7) }
        words += i(71, 12, 11, 7)
        if flat { words += i(71, 12, 14) }
        words += i(71, 13, 30, 0)
        let storage: UInt32 = input ? 1 : 3
        for instruction in [
            i(19, 1), i(21, 2, 32, 1), i(21, 3, 32, 0), i(22, 4, 32), i(23, 5, 3, 4),
            i(32, 6, storage, primitiveType), i(32, 7, 3, 5), i(33, 8, 1),
            i(59, 6, 12, storage), i(59, 7, 13, 3), i(54, 1, 20, 0, 8), i(248, 21),
            i(61, primitiveType, 22, 12),
        ] { words += instruction }
        let colorLane: UInt32
        if primitiveType == 3 {
            colorLane = 22
        } else {
            words += i(124, 3, 23, 22)
            colorLane = 23
        }
        return words + i(80, 5, 24, colorLane, colorLane, colorLane, colorLane) + i(62, 13, 24) + i(253) + i(56)
    }

    @Test(arguments: [UInt32(2), 3], [false, true])
    func spirvPrimitiveIndexNormalizesSignedAndUnsignedInputs(capability: UInt32, signed: Bool) throws {
        let words = Self.primitiveIndexModule(capability: capability, primitiveType: signed ? 2 : 3)
        let compiler = IndependentShaderCompiler()
        let module = try compiler.parseAndValidate(.spirv(words), capabilities: [.primitiveIndex]).module
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        let argument = try #require(entry.arguments.first)
        #expect(argument.binding == .builtIn(.primitiveIndex))
        #expect(try module.types.value(for: argument.type).kind == .scalar(.u32))
        #expect(entry.expressions.values.count(where: { if case .bitcast = $0.kind { true } else { false } }) == (signed ? 1 : 0))
    }

    @Test func spirvPrimitiveIndexRejectsInvalidCapabilityShapeStageAndDirection() throws {
        try fails(Self.primitiveIndexModule(capability: nil), code: .unsupportedCapability)
        try fails(Self.primitiveIndexModule(primitiveType: 4))
        try fails(Self.primitiveIndexModule(model: 0))
        try fails(Self.primitiveIndexModule(input: false))
        try fails(Self.primitiveIndexModule(flat: false))
        #expect(throws: ShaderFailure.self) {
            try IndependentShaderCompiler().parseAndValidate(.spirv(Self.primitiveIndexModule()))
        }
    }

    @Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
    func spirvPrimitiveIndexReachesTextBackends(format: ShaderOutputFormat) throws {
        let compiler = IndependentShaderCompiler()
        let shader = try compiler.compileRenderStage(.spirv(Self.primitiveIndexModule()), entryPoint: "main", to: format)
        #expect(shader.readsPrimitiveIndex)
        if format == .wgsl {
            guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
            _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.primitiveIndex])
        }
    }
}
