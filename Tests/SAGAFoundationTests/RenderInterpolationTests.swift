import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let interpolationProgram = """
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(linear) affine: f32,
        @location(1) smooth: f32,
        @location(2) @interpolate(flat) signed: vec2i,
        @location(3) @interpolate(flat, either) unsigned: u32,
    }
    fn vertex(position: vec4f, value: f32, tag: vec2i, code: u32) -> VertexOutput {
        return VertexOutput(position, value, value, tag, code);
    }
    @vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOutput {
        if index == 0u { return vertex(vec4f(-1, 1, 0.25, 1), 0.0, vec2i(-17, 5), 249u); }
        if index == 1u { return vertex(vec4f(6, 2, 0.5, 2), 1.0, vec2i(1, 7), 11u); }
        return vertex(vec4f(-4, -12, 1, 4), 0.0, vec2i(3, 9), 200u);
    }
    @fragment fn fs(
        @location(0) @interpolate(linear) affine: f32,
        @location(1) smooth: f32,
        @location(2) @interpolate(flat, first) signed: vec2i,
        @location(3) @interpolate(flat, either) unsigned: u32,
    ) -> @location(0) vec4u {
        return vec4u(select(0u, 21u, affine > 0.24 && affine < 0.26),
                     select(0u, 42u, smooth > 0.18 && smooth < 0.19),
                     select(0u, 63u, signed.x == -17 && signed.y == 5),
                     select(0u, 84u, unsigned == 249u || unsigned == 200u));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsLinearAndFlatVaryingsWithTheirScalarTypes(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(interpolationProgram)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: format)
    if format == .glsl {
        #expect(throws: ShaderFailure.unsupported("Render code generation: GLSL flat first sampling")) {
            try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
        }
        return
    }
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
    #expect(vertex.varyings == fragment.varyings)
    #expect(fragment.varyings.map(\.interpolation) == [.linear, .perspective, .flat, .flat])
    #expect(fragment.varyings.map(\.sampling) == [.center, .center, .first, .either])
    #expect(fragment.varyings.map(\.scalar) == [.f32, .f32, .i32, .u32])
    guard case .text(_, let vs) = vertex.artifact, case .text(_, let fs) = fragment.artifact else {
        Issue.record("Expected source"); return
    }
    switch format {
    case .wgsl:
        #expect(fs.contains("@interpolate(linear)") && fs.contains("@interpolate(flat, either)"))
        for body in [vs, fs] {
            _ = try compiler.parseAndValidate(.wgsl(body), capabilities: .referenceDefault.union(.linearInterpolation))
        }
    case .msl:
        #expect(fs.contains("center_no_perspective") && fs.contains(", flat"))
        #expect(vs.contains("int2 saga_member_3"))
    case .hlsl:
        #expect(fs.contains("noperspective float") && fs.contains("nointerpolation int2"))
        #expect(vs.contains("int2 arg3") && vs.contains("uint arg4"))
    case .glsl:
        #expect(vs.contains("noperspective out float") && fs.contains("noperspective in float"))
        #expect(vs.contains("flat out ivec2") && fs.contains("flat in ivec2"))
    default: Issue.record("Unexpected shader target")
    }
}

#if canImport(Metal)
@Test func interpolatesLinearPerspectiveAndFlatVaryingsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let source = ShaderInput.wgsl(interpolationProgram)
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
        let mismatch = ShaderInput.wgsl(interpolationProgram.replacingOccurrences(
            of: "@location(0) @interpolate(linear) affine: f32,", with: "@location(0) affine: f32,"))
        let wrongFragment = try compiler.compileRenderStage(mismatch, entryPoint: "fs", to: .msl)
        await #expect(throws: GPUError.validation("Fragment inputs do not match vertex outputs")) {
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: wrongFragment, colorFormat: .rgba8Uint)
        }
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
            usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 0), vertexCount: 3),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        ])
        #expect(try await ticket.wait() == [[0x543f2a15]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
