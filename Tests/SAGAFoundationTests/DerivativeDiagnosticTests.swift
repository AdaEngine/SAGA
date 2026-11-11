import Testing
import SAGACompiler
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation
import SAGATypes
import SAGACore
import SAGAMetal

private let derivativeDiagnosticShader = """
    diagnostic(off, derivative_uniformity);
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @diagnostic(info, derivative_uniformity)
    fn helper(x: f32) -> f32 {
        if x < 0.0 { return dpdx(x); }
        return 0.0;
    }
    @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
        let red = helper(p.x);
        if p.y < 0.0 { return vec4f(dpdy(p.x)); }
        return vec4f(red, 1, 0, 1);
    }
    """

@Test func spirvDerivativeCompatibilityIsExplicitAndEmitsWGSLDirective() throws {
    let strict = IndependentShaderCompiler()
    let input = ShaderInput.wgsl("""
        diagnostic(off, derivative_uniformity);
        @fragment fn main(@location(0) x: f32) -> @location(0) f32 {
            if x > 0.0 { return dpdx(x); }
            return 0.0;
        }
        """)
    guard case .spirv(let words) = try strict.compile(input, to: .spirv) else {
        Issue.record("Expected SPIR-V")
        return
    }
    #expect(throws: ShaderFailure.self) {
        try strict.compileWebGPURenderStage(.spirv(words), entryPoint: "main")
    }
    let compatible = IndependentShaderCompiler(spirvOptions: .init(allowNonUniformDerivatives: true))
    let result = try compatible.compileWebGPURenderStage(.spirv(words), entryPoint: "main")
    guard case .text(.wgsl, let source) = result.artifact else {
        Issue.record("Expected WGSL")
        return
    }
    #expect(source.contains("diagnostic(off, derivative_uniformity);"))
    #expect(source.contains("dpdx("))
    _ = try strict.parseAndValidate(.wgsl(source))
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsFragmentDiagnosticFiltersOnlyInWGSL(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let fragment = try compiler.compileRenderStage(.wgsl(derivativeDiagnosticShader), entryPoint: "fs", to: format)
    guard case .text(_, let source) = fragment.artifact else { Issue.record("Expected shader source"); return }
    if format == .wgsl {
        #expect(source.hasPrefix("diagnostic(off, derivative_uniformity);"))
        #expect(source.contains("@diagnostic(info, derivative_uniformity)"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    } else {
        #expect(!source.contains("derivative_uniformity"))
    }
}

@Test(arguments: [
    ("diagnostic(off, derivative_uniformity); @diagnostic(error, derivative_uniformity) @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f { if x > 0.0 { return vec4f(dpdx(x)); } return vec4f(0); }", ShaderDiagnosticCode.nonUniformControlFlow),
    ("diagnostic(off, derivative_uniformity); @compute @workgroup_size(1) fn run() { let x = dpdx(0.0); }", .invalidControlFlow),
    ("diagnostic(off, derivative_uniformity); @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) i: u32) { if i > 0u { workgroupBarrier(); } }", .nonUniformControlFlow),
    ("fn strict(x: f32) -> f32 { return dpdx(x); } @diagnostic(off, derivative_uniformity) @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f { if x > 0.0 { return vec4f(strict(x)); } return vec4f(0); }", .nonUniformControlFlow),
    ("diagnostic(off, derivative_uniformity); diagnostic(error, derivative_uniformity);", .duplicateName),
    ("@diagnostic(off, derivative_uniformity) @diagnostic(off, derivative_uniformity) fn f() {}", .duplicateName),
    ("diagnostic(warn, derivative_uniformity);", .unexpectedToken),
])
func diagnosticsPreserveStrictDerivativeAndBarrierChecks(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid diagnostic directive or filtered shader accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

@Test func calleeFilterControlsDerivativeRequirementAcrossCalls() throws {
    let source = """
        @diagnostic(off, derivative_uniformity) fn relaxed(x: f32) -> f32 { return dpdx(x); }
        @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
            if x > 0.0 { return vec4f(relaxed(x)); }
            return vec4f(0);
        }
        """
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
}

@Test func reportsFilteredDiagnosticsWithSeverityAndSourceLocation() throws {
    let compiler = IndependentShaderCompiler()
    let source = """
        diagnostic(off, derivative_uniformity);
        diagnostic(off, derivative_uniformity);
        diagnostic(info, vendor.rule);
        @diagnostic(warning, derivative_uniformity) @diagnostic(off, vendor.other)
        fn helper(x: f32) -> f32 { return dpdx(x); }
        @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
            if x > 0.0 { return vec4f(helper(x)); }
            return vec4f(0);
        }
        """
    let validated = try compiler.parseAndValidate(.wgsl(source))
    let warning = try #require(validated.diagnostics.first)
    #expect(validated.diagnostics.count == 1 && warning.severity == .warning && warning.code == .nonUniformControlFlow)
    let span = try #require(warning.span)
    #expect(String(decoding: Array(source.utf8)[span.startUTF8..<span.endUTF8], as: UTF8.self) == "dpdx")
    let info = try compiler.parseAndValidate(.wgsl(derivativeDiagnosticShader)).diagnostics
    #expect(info.count == 1 && info.first?.severity == .info)
    let unknown = try compiler.parseAndValidate(.wgsl("diagnostic(off, unknown_rule);"))
    #expect(unknown.diagnostics.map(\.code) == [.unknownDiagnosticRule])
}

@Test func rawDiagnosticFiltersRequireValidParentsAndRuleNames() throws {
    let cycle = IRDiagnosticFilter(parent: try .init(index: 0), rule: .derivativeUniformity, severity: .off)
    let invalidName = IRDiagnosticFilter(rule: .unknown("foo);"), severity: .off)
    for (filter, code) in [(cycle, ShaderDiagnosticCode.invalidReference), (invalidName, .invalidType)] {
        do {
            try ShaderSemanticValidator().validate(IRModule(diagnosticFilters: IRTable([filter])))
            Issue.record("Invalid diagnostic filter IR accepted")
        } catch ShaderFailure.diagnostics(let issues) {
            #expect(issues.first?.code == code)
        }
    }
}

@Test func derivativeFilterAlsoAppliesToImplicitTextureSampling() throws {
    let source = """
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var state: sampler;
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            if uv.x > 0.0 { return textureSampleBias(image, state, uv, 0.5); }
            return vec4f(0);
        }
        """
    let compiler = IndependentShaderCompiler()
    do {
        _ = try compiler.parseAndValidate(.wgsl(source))
        Issue.record("Implicit texture sampling in divergent control flow accepted by default")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .nonUniformControlFlow)
    }
    let filtered = try compiler.parseAndValidate(.wgsl("diagnostic(off, derivative_uniformity);\n" + source))
    #expect(filtered.diagnostics.isEmpty)
}

#if canImport(Metal)
@Test func runsShaderWithDerivativeDiagnosticsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(derivativeDiagnosticShader)
        let pipeline = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl), colorFormat: .rgba8Unorm)
        let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 264, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: texture, clear: .transparentBlack, vertexCount: 3),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 264),
        ])
        let pixels = try #require(try await ticket.wait().first)
        #expect([0, 1, 64, 65].map { pixels[$0] } == Array(repeating: 0xff00ff00, count: 4))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
