import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

private let writableSwizzleFragment = ShaderInput.glsl("""
    #version 450 core
    layout(location = 0) out vec4 color;
    void main() {
        color = vec4(0.0, 0.5, 1.0, 1.0);
        color.rgb = color.bgr;
        color.xy += color.yy;
        color.xy *= 0.5;
        color.z = 1.0;
    }
    """, stage: .fragment, defines: [:])

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func glslWritableSwizzlesReachTextTargets(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let result = try compiler.compileRenderStage(writableSwizzleFragment, entryPoint: "main", to: format)
    #expect(result.colorOutputComponentCounts == [0: 4])
    if format == .wgsl, case .text(.wgsl, let text) = result.artifact {
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

@Test(arguments: [
    ("color.xx = vec2(1.0);", ShaderDiagnosticCode.invalidAddressSpace),
    ("value.xy = vec2(1.0);", .invalidAddressSpace),
    ("value.x = 1.0;", .invalidAddressSpace),
    ("color.xy = 1.0;", .typeMismatch),
])
func glslWritableSwizzlesRejectRepeatedLanesReadonlyAndScalarAssignments(
    _ assignment: String, _ expected: ShaderDiagnosticCode
) throws {
    let input = ShaderInput.glsl("""
        #version 450 core
        layout(location = 0) in vec4 value;
        layout(location = 0) out vec4 color;
        void main() { \(assignment) }
        """, stage: .fragment, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(input)
        Issue.record("Invalid GLSL swizzle assignment accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == expected)
    }
}

#if canImport(Metal)
@Test func glslWritableSwizzlesSnapshotOverlappingLanesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl("""
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0, 1); }
            if index == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        """), entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(writableSwizzleFragment, entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: target, colorLoad: .clear(.init(red: 0, green: 0, blue: 0, alpha: 0)),
                    vertexCount: 3),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ])
        #expect(try await ticket.wait() == [[0xffff80bf]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
