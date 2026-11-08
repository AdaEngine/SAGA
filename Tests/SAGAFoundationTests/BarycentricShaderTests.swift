import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let barycentricShader = """
    struct AffineInput { @builtin(barycentric_no_perspective) weights: vec3f, }
    @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        let points = array(vec4f(-1, -1, 0, 1), vec4f(2, -2, 0, 2), vec4f(-4, 4, 0, 4));
        return points[index];
    }
    @fragment fn perspective(@builtin(barycentric) weights: vec3f) -> @location(0) vec4f {
        return vec4f(weights, 1);
    }
    @fragment fn perspective_with_position(
        @builtin(barycentric) weights: vec3f, @builtin(position) pixel: vec4f
    ) -> @location(0) vec4f {
        return vec4f(weights, pixel.w);
    }
    @fragment fn affine(input: AffineInput) -> @location(0) vec4f {
        return vec4f(input.weights, 1);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func barycentricShaderTranslatesBothInterpolationModes(_ format: ShaderOutputFormat, _ perspective: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let entry = perspective ? "perspective" : "affine"
    let shader = try compiler.compileRenderStage(.wgsl(barycentricShader), entryPoint: entry, to: format)
    #expect(shader.requiresBarycentrics)
    guard case .text(let actual, let text) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    switch format {
    case .msl:
        let attribute = perspective ? "[[barycentric_coord]]" : "[[barycentric_coord, center_no_perspective]]"
        #expect(text.contains(attribute))
    case .hlsl:
        #expect(shader.hlslProfile == "ps_6_1")
        #expect(text.contains("SV_Barycentrics"))
        #expect(text.contains("noperspective float3") == !perspective)
    case .glsl:
        #expect(text.contains("#extension GL_EXT_fragment_shader_barycentric : require"))
        #expect(text.contains(perspective ? "gl_BaryCoordEXT" : "gl_BaryCoordNoPerspEXT"))
    case .wgsl:
        #expect(text.contains(perspective ? "@builtin(barycentric)" : "@builtin(barycentric_no_perspective)"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.shaderBarycentrics])
    default: Issue.record("Expected a text target")
    }
}

@Test func barycentricShaderRequiresFragmentInputShapeAndCapability() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(barycentricShader)) }
    _ = try compiler.parseAndValidate(.wgsl(barycentricShader), capabilities: [.shaderBarycentrics])
    for source in [
        "@fragment fn f(@builtin(barycentric) b: vec2f) {}",
        "@fragment fn f(@builtin(barycentric) b: vec3u) {}",
        "@vertex fn f(@builtin(barycentric) b: vec3f) -> @builtin(position) vec4f { return vec4f(b, 1); }",
        "@fragment fn f() -> @builtin(barycentric) vec3f { return vec3f(0); }",
        "@fragment fn f(@builtin(barycentric) a: vec3f, @builtin(barycentric_no_perspective) b: vec3f) {}",
    ] {
        #expect(throws: ShaderFailure.self) {
            try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderBarycentrics])
        }
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileWebGPURenderStage(.wgsl(barycentricShader), entryPoint: "perspective")
    }
    let vertex = try compiler.compileWebGPURenderStage(.wgsl(barycentricShader), entryPoint: "vs")
    #expect(!vertex.requiresBarycentrics)
}

private let glslBarycentricExtension = "GL_EXT_fragment_shader_barycentric"

private func glslBarycentricShader(perspective: Bool, directives: String? = nil, version: UInt32 = 450) -> String {
    let builtin = perspective ? "gl_BaryCoordEXT" : "gl_BaryCoordNoPerspEXT"
    return """
        #version \(version) core
        \(directives ?? "#extension \(glslBarycentricExtension) : require")
        layout(location = 0) out vec4 color;
        vec4 readWeights() { return vec4(\(builtin), 1.0); }
        void main() { color = readWeights(); }
        """
}

extension SPIRVFrontendTests {
    static func barycentricModule(
        perspective: Bool = true, capability: Bool = true, extensionDeclared: Bool = true,
        model: UInt32 = 4, width: UInt32 = 3, writeInput: Bool = false
    ) -> [UInt32] {
        var words = head + i(17, 1)
        if capability { words += i(17, 5284) }
        if extensionDeclared {
            var string = Array("SPV_KHR_fragment_shader_barycentric".utf8) + [0]
            while string.count % 4 != 0 { string.append(0) }
            words.append(UInt32(string.count / 4 + 1) << 16 | 10)
            for start in stride(from: 0, to: string.count, by: 4) {
                let word = UInt32(string[start]) | UInt32(string[start + 1]) << 8
                let high = UInt32(string[start + 2]) << 16 | UInt32(string[start + 3]) << 24
                words.append(word | high)
            }
        }
        for instruction in [
            i(14, 0, 1), i(15, model, 20, 0x6e69616d, 0, 16, 17), model == 4 ? i(16, 20, 7) : [],
            i(71, 16, 11, perspective ? 5286 : 5287), i(71, 17, 30, 0),
            i(19, 1), i(22, 2, 32), i(23, 3, 2, width), i(23, 4, 2, 4),
            i(32, 5, 1, 3), i(32, 6, 3, 4), i(33, 7, 1), i(33, 8, 4),
            i(43, 2, 10, 0x3f800000), i(59, 5, 16, 1), i(59, 6, 17, 3),
            i(54, 1, 20, 0, 7), i(248, 21), i(57, 4, 22, 30), i(62, 17, 22), i(253), i(56),
            i(54, 4, 30, 0, 8), i(248, 31), i(61, 3, 32, 16), writeInput ? i(62, 16, 32) : [],
            i(81, 2, 33, 32, 0), i(81, 2, 34, 32, 1), i(81, 2, 35, 32, 2),
            i(80, 4, 36, 33, 34, 35, 10), i(254, 36), i(56),
        ] { words += instruction }
        return words
    }

    @Test func barycentricImportsCheckSPIRVDeclarationsAndInterface() throws {
        try fails(Self.barycentricModule(capability: false), code: .unsupportedCapability)
        try fails(Self.barycentricModule(extensionDeclared: false), code: .unsupportedCapability)
        try fails(Self.barycentricModule(model: 0))
        try fails(Self.barycentricModule(width: 4))
        try fails(Self.barycentricModule(writeInput: true))
    }
}

@Test func barycentricImportsCheckGLSLExtensionAtEachUse() throws {
    let compiler = IndependentShaderCompiler()
    let feature = glslBarycentricExtension
    let enabled = """
        #if !defined(\(feature))
        #error Missing extension support macro
        #endif
        #extension \(feature) : enable
        """
    let disabled = "#extension \(feature) : disable"
    func parse(_ source: String, stage: ShaderStage = .fragment) throws {
        _ = try compiler.parseAndValidate(.glsl(source, stage: stage, defines: [:]), capabilities: [.shaderBarycentrics])
    }
    try parse(glslBarycentricShader(perspective: true, directives: enabled))
    try parse(glslBarycentricShader(perspective: false, directives: "#extension \(feature) : warn"))
    for directives in ["", disabled, "#if 0\n\(enabled)\n#endif", enabled + "\n" + disabled] {
        #expect(throws: ShaderFailure.self) { try parse(glslBarycentricShader(perspective: true, directives: directives)) }
    }
    let changed = """
        #version 450 core
        \(enabled)
        layout(location = 0) out vec4 color;
        vec3 readWeights() { return gl_BaryCoordEXT; }
        \(disabled)
        void main() { color = vec4(gl_BaryCoordEXT + readWeights(), 1.0); }
        """
    #expect(throws: ShaderFailure.self) { try parse(changed) }
    #expect(throws: ShaderFailure.self) { try parse(glslBarycentricShader(perspective: true, version: 440)) }
    #expect(throws: ShaderFailure.self) {
        try parse("#version 450\n\(enabled)\nvoid main() { gl_Position = vec4(gl_BaryCoordEXT, 1.0); }", stage: .vertex)
    }
    #expect(throws: ShaderFailure.self) {
        try parse("#version 450\n\(enabled)\nvoid main() { gl_BaryCoordEXT = vec3(0.0); }")
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl], [false, true])
func barycentricImportsReachAllTextTargets(_ format: ShaderOutputFormat, _ perspective: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let inputs: [ShaderInput] = [
        .glsl(glslBarycentricShader(perspective: perspective), stage: .fragment, defines: [:]),
        .spirv(SPIRVFrontendTests.barycentricModule(perspective: perspective)),
    ]
    for input in inputs {
        let module = try compiler.parseAndValidate(input, capabilities: [.shaderBarycentrics]).module
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        #expect(entry.arguments.first?.binding == .builtIn(.barycentric(perspective: perspective)))
        let fragment = try compiler.compileRenderStage(input, entryPoint: "main", to: format)
        #expect(fragment.requiresBarycentrics)
        guard case .text(let target, let text) = fragment.artifact else { throw ShaderFailure.invalidHandle }
        #expect(target == format)
        if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.shaderBarycentrics]) }
        if format == .glsl {
            _ = try compiler.parseAndValidate(.glsl(text, stage: .fragment, defines: [:]), capabilities: [.shaderBarycentrics])
        }
        #expect(throws: ShaderFailure.self) { try compiler.compileWebGPURenderStage(input, entryPoint: "main") }
    }
}

#if canImport(Metal)
@Test(arguments: [ShaderInputFormat.wgsl, .glsl, .spirv])
func barycentricShaderDistinguishesPerspectiveAndAffineWeightsOnMetal(_ inputFormat: ShaderInputFormat) async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(barycentricShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderBarycentrics])
    do {
        let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba32Float,
                                                            usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
        let perspective: [Float] = [8.0 / 11, 2.0 / 11, 1.0 / 11]
        let affine: [Float] = [0.5, 0.25, 0.25, 1]
        let modes: [(ShaderInput, String, [Float])]
        switch inputFormat {
        case .wgsl:
            modes = [(source, "affine", affine), (source, "perspective", perspective + [1]),
                     (source, "perspective_with_position", perspective + [11.0 / 16])]
        case .glsl:
            modes = [
                (.glsl(glslBarycentricShader(perspective: false), stage: .fragment, defines: [:]), "main", affine),
                (.glsl(glslBarycentricShader(perspective: true), stage: .fragment, defines: [:]), "main", perspective + [1]),
            ]
        case .spirv:
            modes = [
                (.spirv(SPIRVFrontendTests.barycentricModule(perspective: false)), "main", affine),
                (.spirv(SPIRVFrontendTests.barycentricModule()), "main", perspective + [1]),
            ]
        }
        for (input, entry, expected) in modes {
            let fragment = try compiler.compileRenderStage(input, entryPoint: entry, to: .msl)
            let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba32Float)
            let snapshots = try await device.submitCommands([
                .render(pipeline: pipeline, target: texture, clear: .transparentBlack, vertexCount: 3),
                .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
                .read(buffer: readback, byteCount: 512),
            ]).wait()
            let pixels = try #require(snapshots.first)
            try #require(pixels.count == 128)
            for (word, value) in zip(pixels[64..<68], expected) {
                #expect(abs(Float(bitPattern: word) - value) < 0.001)
            }
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
