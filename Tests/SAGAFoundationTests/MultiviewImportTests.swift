import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal
#endif

private func multiviewGLSL(_ stage: ShaderStage, directives: String = "#extension GL_EXT_multiview : require") -> ShaderInput {
    let output = stage == .vertex ? "gl_Position" : "color"
    let declaration = stage == .vertex ? "" : "layout(location = 0) out vec4 color;"
    return .glsl("""
        #version 450 core
        \(directives)
        \(declaration)
        float signedView() { return float(gl_ViewIndex - 1); }
        void main() { \(output) = vec4(signedView(), 0.0, 0.0, 1.0); }
        """, stage: stage, defines: [:])
}

extension SPIRVFrontendTests {
    static func viewIndexModule(
        model: UInt32 = 4, signed: Bool = true, version: UInt32 = 0x00010300,
        capability: Bool = true, extensionDeclared: Bool = false, width: UInt32 = 32, writeInput: Bool = false
    ) -> [UInt32] {
        var words = head
        words[1] = version
        words += i(17, 1)
        if capability { words += i(17, 4439) }
        if width == 64 { words += i(17, 11) }
        if extensionDeclared {
            // SPV_KHR_multiview, encoded as little-endian SPIR-V string operands.
            words += i(10, 0x5f565053, 0x5f52484b, 0x746c756d, 0x65697669, 0x00000077)
        }
        for instruction in [
            i(14, 0, 1), i(15, model, 20, 0x6e69616d, 0, 16, 17),
            model == 4 ? i(16, 20, 7) : model == 5 ? i(16, 20, 17, 1, 1, 1) : [],
            i(71, 16, 11, 4440), model == 4 ? i(71, 16, 14) : [],
            model == 0 ? i(71, 17, 11, 0) : i(71, 17, 30, 0),
            i(19, 1), i(21, 2, width, signed ? 1 : 0), i(22, 3, 32), i(23, 4, 3, 4),
            i(32, 5, 1, 2), i(32, 6, 3, 4), i(33, 7, 1),
            i(43, 3, 10, 0), i(43, 3, 11, 0x3f800000),
            width == 64 ? i(43, 2, 12, 1, 0) : i(43, 2, 12, 1), i(59, 5, 16, 1), i(59, 6, 17, 3),
            i(54, 1, 20, 0, 7), i(248, 21), i(61, 2, 22, 16), writeInput ? i(62, 16, 22) : [],
            i(130, 2, 23, 22, 12), i(signed ? 111 : 112, 3, 24, 23), i(80, 4, 25, 24, 10, 10, 11),
            i(62, 17, 25), i(253), i(56),
        ] { words += instruction }
        return words
    }

    @Test func multiviewImportChecksSPIRVVersionCapabilityAndInput() throws {
        _ = try parse(Self.viewIndexModule(version: 0x00010000, extensionDeclared: true))
        try fails(Self.viewIndexModule(version: 0x00010200), code: .unsupportedCapability)
        try fails(Self.viewIndexModule(capability: false), code: .unsupportedCapability)
        try fails(Self.viewIndexModule(model: 5))
        try fails(Self.viewIndexModule(width: 64))
        try fails(Self.viewIndexModule(writeInput: true))
    }

    @Test(arguments: [UInt32(111), 112])
    func multiviewImportIntegerToFloatUsesOpcodeSignedness(_ opcode: UInt32) throws {
        let declaredSigned: UInt32 = opcode == 111 ? 0 : 1
        var words = Self.pre + Self.i(15, 5, 20, 0x6e69616d, 0) + Self.i(16, 20, 17, 1, 1, 1)
        for instruction in [
            Self.i(19, 1), Self.i(21, 2, 32, declaredSigned), Self.i(22, 3, 32),
            Self.i(23, 4, 2, 2), Self.i(23, 5, 3, 2), Self.i(33, 6, 1),
            Self.i(43, 2, 7, .max), Self.i(44, 4, 8, 7, 7),
            Self.i(54, 1, 20, 0, 6), Self.i(248, 21), Self.i(opcode, 5, 22, 8), Self.i(253), Self.i(56),
        ] { words += instruction }
        let module = try parse(words)
        let expressions = module.functions.values[0].expressions
        let expected: IRScalar = opcode == 111 ? .i32 : .u32
        let cast = try #require(expressions.values.first {
            if case .cast(_, .f32) = $0.kind { return true }
            return false
        })
        guard case .cast(let input, _) = cast.kind else { throw ShaderFailure.invalidHandle }
        let castOperand = try expressions.value(for: input)
        #expect(try module.types.value(for: castOperand.type).kind == .vector(size: .two, scalar: expected))
        let conversion = try #require(words.firstIndex(of: 4 << 16 | opcode))
        words[conversion + 1] = 3
        try fails(words)
    }
}

@Test(arguments: [ShaderStage.vertex, .fragment], [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func multiviewImportReachesAllTextTargets(_ stage: ShaderStage, _ output: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let model: UInt32 = stage == .vertex ? 0 : 4
    let inputs = [multiviewGLSL(stage), .spirv(SPIRVFrontendTests.viewIndexModule(model: model)),
                  .spirv(SPIRVFrontendTests.viewIndexModule(model: model, signed: false))]
    for (index, input) in inputs.enumerated() {
        let module = try compiler.parseAndValidate(input, capabilities: [.multiview]).module
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        let parameter = try #require(entry.arguments.first)
        #expect(parameter.binding == .builtIn(.viewIndex))
        #expect(try module.types.value(for: parameter.type).kind == .scalar(.u32))
        let artifact = try compiler.compileRenderStage(input, entryPoint: "main", to: output)
        #expect(artifact.readsViewIndex)
        guard case .text(_, let code) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
        switch output {
        case .wgsl:
            #expect(code.contains("bitcast<i32>") == (index != 2))
            _ = try compiler.parseAndValidate(.wgsl(code), capabilities: [.multiview])
        case .glsl:
            #expect(code.contains("uint(gl_ViewIndex)"))
            _ = try compiler.parseAndValidate(.glsl(code, stage: stage, defines: [:]), capabilities: [.multiview])
        case .hlsl:
            #expect(code.contains("SV_ViewID"))
            #expect(artifact.hlslProfile == (stage == .vertex ? "vs_6_1" : "ps_6_1"))
        case .msl: #expect(code.contains("[[amplification_id]]"))
        default: Issue.record("Expected a text target")
        }
        #expect(throws: ShaderFailure.unsupported("Render code generation: WebGPU multiview index")) {
            try compiler.compileWebGPURenderStage(input, entryPoint: "main")
        }
    }
}

@Test func multiviewImportRespectsGLSLExtensionScopeAndReadOnlyInput() throws {
    let compiler = IndependentShaderCompiler()
    let enable = "#extension GL_EXT_multiview : enable"
    let disable = "#extension GL_EXT_multiview : disable"
    let macro = """
        #if !defined(GL_EXT_multiview)
        #error Missing extension support macro
        #endif
        #define VIEW gl_ViewIndex
        \(enable)
        """
    let input = ShaderInput.glsl("""
        #version 450
        \(macro)
        layout(location = 0) out vec4 color;
        void main() { color = vec4(float(VIEW)); }
        """, stage: .fragment, defines: [:])
    _ = try compiler.parseAndValidate(input, capabilities: [.multiview])
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(input, capabilities: []) }
    for directives in ["", disable, "#if 0\n\(enable)\n#endif", "\(enable)\n\(disable)"] {
        #expect(throws: ShaderFailure.self) {
            try compiler.parseAndValidate(multiviewGLSL(.fragment, directives: directives), capabilities: [.multiview])
        }
    }
    for source in [
        "#version 450\n\(enable)\nvoid main() { gl_ViewIndex = 2; }",
        "#version 450\n\(enable)\nvoid main() { int gl_ViewIndex = 2; }",
        "#version 450\n\(macro)\nint readView() { return VIEW; }\n\(disable)\nvoid main() { int view = VIEW; }",
    ] {
        #expect(throws: ShaderFailure.self) {
            try compiler.parseAndValidate(.glsl(source, stage: .fragment, defines: [:]), capabilities: [.multiview])
        }
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.glsl("""
            #version 450
            \(enable)
            layout(local_size_x = 1) in;
            void main() { int index = gl_ViewIndex; }
            """, stage: .compute, defines: [:]), capabilities: [.multiview])
    }
}

#if canImport(Metal)
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsVertexAmplificationCount(4) == true),
      arguments: [ShaderInputFormat.glsl, .spirv])
func multiviewImportRendersSparseLayersOnMetal(_ format: ShaderInputFormat) async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return vec4f(points[index], 0, 1);
        }
        """)
    let input = format == .glsl ? multiviewGLSL(.fragment) : .spirv(SPIRVFrontendTests.viewIndexModule())
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.multiview, .selectiveMultiview])
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment,
            colorFormat: .rgba32Float, multiviewMask: 0b101)
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba32Float,
            usage: [.renderAttachment, .copySource], arrayLayerCount: 3))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = [.render(pipeline: pipeline, target: texture, clear: .transparentBlack, vertexCount: 3)]
        for layer: UInt32 in 0..<3 {
            commands += [.copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, arrayLayer: layer),
                         .read(buffer: read, byteCount: 16)]
        }
        let expected: [[Float]] = [[-1, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 1]]
        #expect(try await device.submitCommands(commands).wait() == expected.map { $0.map(\.bitPattern) })
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
