import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let derivativeShader = """
    struct Outputs {
        @location(0) regular: vec4u,
        @location(1) fine: vec4u,
        @location(2) coarse: vec4u,
    }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    fn encode(dx: f32, dy: f32, width: vec2f) -> vec4u {
        return vec4u(u32(dx), u32(8.0 + dy), u32(width.x), u32(width.y));
    }
    fn derivatives(p: vec2f) -> Outputs {
        let q = vec2f(2.0 * p.x - 3.0 * p.y, 5.0 * p.y - 4.0 * p.x);
        let regular = encode(dpdx(q.x), dpdy(q.x), fwidth(q));
        let fine = encode(dpdxFine(q).x, dpdyFine(q).x, fwidthFine(q));
        let coarse = encode(dpdxCoarse(q).x, dpdyCoarse(q).x, fwidthCoarse(q));
        if p.x > 0.0 { return Outputs(regular, fine, coarse); }
        return Outputs(vec4u(0), vec4u(0), vec4u(0));
    }
    @fragment fn fs(@builtin(position) p: vec4f) -> Outputs { return derivatives(p.xy); }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsFragmentDerivativeVariants(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileRenderStage(.wgsl(derivativeShader), entryPoint: "fs", to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("dpdxFine(") && source.contains("dpdyCoarse(") && source.contains("fwidthFine("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains("dfdx(") && source.contains("dfdy(") && source.contains("fwidth("))
    case .hlsl:
        #expect(source.contains("ddx_fine(") && source.contains("ddy_coarse(") && source.contains("abs(ddx_coarse("))
    case .glsl:
        #expect(source.contains("dFdxFine(") && source.contains("dFdyCoarse(") && source.contains("fwidthFine("))
    default: Issue.record("Unexpected shader target")
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func evaluatesDerivativeBeforeVaryingBranch(format: ShaderOutputFormat) throws {
    let source = """
        @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
            let slope = dpdx(x);
            if x > 0.0 { return vec4f(slope); }
            return vec4f(0.0);
        }
        """
    let shader = try IndependentShaderCompiler().compileRenderStage(.wgsl(source), entryPoint: "fs", to: format)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected shader source"); return }
    let lines = body.split(separator: "\n")
    let derivative = try #require(lines.firstIndex(where: { $0.contains("saga_derivative_") && $0.contains(" = ") }))
    let branch = try #require(lines.firstIndex(where: { $0.contains("if (") }))
    #expect(derivative < branch)
}

@Test(arguments: [
    ("@compute @workgroup_size(1) fn run() { let x = dpdx(1.0); }", ShaderDiagnosticCode.invalidControlFlow),
    ("fn slope() -> f32 { return dpdyFine(1.0); } @vertex fn vs() -> @builtin(position) vec4f { return vec4f(slope()); }", .invalidControlFlow),
    ("@fragment fn fs(@location(0) x: f32) -> @location(0) f32 { if x > 0.0 { return dpdx(x); } return 0.0; }", .nonUniformControlFlow),
    ("fn slope(x: f32) -> f32 { return fwidth(x); } @fragment fn fs(@location(0) x: f32) -> @location(0) f32 { if x > 0.0 { return slope(x); } return 0.0; }", .nonUniformControlFlow),
    ("@fragment fn fs(@location(0) x: f32) -> @location(0) f32 { if x > 0.0 { return 0.0; } return dpdy(x); }", .nonUniformControlFlow),
    ("@fragment fn fs() -> @location(0) f32 { let x = dpdyCoarse(vec2u(1)); return 0.0; }", .typeMismatch),
    ("@fragment fn fs() -> @location(0) f32 { return dpdx(1.0, 2.0); }", .typeMismatch),
])
func rejectsDerivativeTypeStageAndUniformityErrors(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid derivative accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

#if canImport(Metal)
@Test func computesFragmentDerivativeVariantsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(derivativeShader)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Uint, additionalColorTargets: [1: .init(format: .rgba8Uint), 2: .init(format: .rgba8Uint)])
        let descriptor = TextureDescriptor(width: 2, height: 2, format: .rgba8Uint, usage: [.renderAttachment, .copySource])
        let regular = try await device.createTexture(descriptor)
        let fine = try await device.createTexture(descriptor)
        let coarse = try await device.createTexture(descriptor)
        let read = try await device.createBuffer(.init(size: 264, usage: [.mapRead, .copyDestination]))
        var commands: [BufferCommand] = [
            .render(pipeline: pipeline, target: regular, clear: .transparentBlack, vertexCount: 3,
                    additionalColors: [1: .init(texture: fine, loadAction: .clear(.transparentBlack)),
                                       2: .init(texture: coarse, loadAction: .clear(.transparentBlack))]),
        ]
        for texture in [regular, fine, coarse] {
            commands += [.copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 264)]
        }
        let ticket = try await device.submitCommands(commands)
        let pixels = try await ticket.wait().flatMap { snapshot in [0, 1, 64, 65].map { snapshot[$0] } }
        #expect(pixels == Array(repeating: 0x09050502, count: 12))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
