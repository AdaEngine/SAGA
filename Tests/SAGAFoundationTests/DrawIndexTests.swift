import SAGACompiler
import SAGAShaderIR
import Testing

private let drawIndexShader = ShaderInput.wgsl("""
    enable draw_index;
    struct VertexInput { @builtin(draw_index) draw: u32, @builtin(vertex_index) vertex: u32 }
    @vertex fn vs(input: VertexInput) -> @builtin(position) vec4f {
        return vec4f(f32(input.draw), f32(input.vertex), 0.0, 1.0);
    }
    """)

@Test func drawIndexCompilesAndRoundTripsThroughNativeWGSLAndSPIRV() throws {
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(drawIndexShader, capabilities: [.drawIndex])
    guard case .spirv(let words) = try compiler.compile(drawIndexShader, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    for input in [drawIndexShader, .spirv(words)] {
        let artifact = try compiler.compileRenderStage(input, entryPoint: "vs", to: .wgsl)
        guard case .text(.wgsl, let text) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(text.contains("enable draw_index;") && text.contains("@builtin(draw_index)"))
        #expect(text.contains("@builtin(vertex_index)"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.drawIndex])
    }
    let module = try compiler.parseAndValidate(.spirv(words), capabilities: [.drawIndex]).module
    let entry = try module.functions.value(for: #require(module.entryPoints.first).function)
    #expect(entry.arguments.contains { $0.binding == .builtIn(.drawIndex) })
}

@Test func drawIndexRejectsMissingDirectiveCapabilityAndInvalidStageOrType() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(drawIndexShader) }
    for source in [
        "@vertex fn vs(@builtin(draw_index) d: u32) -> @builtin(position) vec4f { return vec4f(f32(d)); }",
        "enable draw_index; @vertex fn vs(@builtin(draw_index) d: i32) -> @builtin(position) vec4f { return vec4f(f32(d)); }",
        "enable draw_index; @fragment fn fs(@builtin(draw_index) d: u32) -> @location(0) vec4f { return vec4f(f32(d)); }",
        "enable draw_index; @vertex fn vs() -> @builtin(draw_index) u32 { return 1u; }",
    ] {
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(source), capabilities: [.drawIndex]) }
    }
}

@Test func drawIndexRejectsUnsupportedTargetsForWGSLAndImportedSPIRV() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(drawIndexShader, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    for input in [drawIndexShader, .spirv(words)] {
        #expect(throws: ShaderFailure.self) { try compiler.compileWebGPURenderStage(input, entryPoint: "vs") }
        for format in [ShaderOutputFormat.msl, .hlsl] {
            #expect(throws: ShaderFailure.self) { try compiler.compileRenderStage(input, entryPoint: "vs", to: format) }
        }
    }
}

extension SPIRVFrontendTests {
    private static func drawIndexModule(
        version: UInt32 = 0x00010300, withCapability: Bool = true, withExtension: Bool = false,
        model: UInt32 = 0, storage: UInt32 = 1, signed: UInt32 = 1, width: UInt32 = 32
    ) -> [UInt32] {
        var words = head
        words[1] = version
        words += i(17, 1)
        if width == 16 { words += i(17, 22) + i(17, 4436) } // Int16 and StorageInputOutput16
        if withCapability { words += i(17, 4427) }
        if withExtension {
            var bytes = Array("SPV_KHR_shader_draw_parameters".utf8)
            bytes += [UInt8](repeating: 0, count: 4 - bytes.count % 4)
            let operands: [UInt32] = stride(from: 0, to: bytes.count, by: 4).map { index in
                let low = UInt32(bytes[index]) | UInt32(bytes[index + 1]) << 8
                let high = UInt32(bytes[index + 2]) << 16 | UInt32(bytes[index + 3]) << 24
                return low | high
            }
            words += [UInt32(operands.count + 1) << 16 | 10] + operands
        }
        words += i(14, 0, 1) + i(15, model, 20, 0x6e69616d, 0, 10, 11)
        if model == 4 { words += i(16, 20, 7) }
        for instruction in [
            i(71, 10, 11, 4426), i(71, 11, model == 4 ? 30 : 11, 0),
            i(19, 1), i(21, 2, width, signed), i(22, 3, 32), i(23, 4, 3, 4),
            i(32, 5, storage, 2), i(32, 6, 3, 4), i(33, 7, 1),
            i(59, 5, 10, storage), i(59, 6, 11, 3), i(54, 1, 20, 0, 7), i(248, 21),
            i(61, 2, 22, 10), i(signed == 1 ? 111 : 112, 3, 23, 22), i(80, 4, 24, 23, 23, 23, 23),
            i(62, 11, 24), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test(arguments: [UInt32(0), 1], [false, true])
    func spirvDrawIndexNormalizesSignedAndLegacyInputs(signed: UInt32, legacy: Bool) throws {
        let words = Self.drawIndexModule(version: legacy ? 0x00010000 : 0x00010300, withExtension: legacy, signed: signed)
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(words), capabilities: [.drawIndex]).module
        let entry = try module.functions.value(for: #require(module.entryPoints.first).function)
        let input = try #require(entry.arguments.first)
        #expect(input.binding == .builtIn(.drawIndex))
        #expect(try module.types.value(for: input.type).kind == .scalar(.u32))
        #expect(entry.expressions.values.contains { if case .bitcast = $0.kind { true } else { false } } == (signed == 1))
    }

    @Test func spirvDrawIndexRequiresCapabilityVersionAndValidInterface() throws {
        try fails(Self.drawIndexModule(withCapability: false), code: .unsupportedCapability)
        try fails(Self.drawIndexModule(version: 0x00010000), code: .unsupportedCapability)
        try fails(Self.drawIndexModule(model: 4))
        try fails(Self.drawIndexModule(storage: 3))
        try fails(Self.drawIndexModule(width: 16))
    }
}
