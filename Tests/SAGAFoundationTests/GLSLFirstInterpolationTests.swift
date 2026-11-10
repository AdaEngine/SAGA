import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes

@Test(arguments: ["first", "either"])
func glslFlatInterpolationRequiresSupportedProvokingVertex(sampling: String) throws {
    let source = ShaderInput.wgsl("""
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(flat, \(sampling)) value: u32,
        }
        @vertex fn vs() -> VertexOutput { return VertexOutput(vec4f(0.0), 1u); }
        @fragment fn fs(@location(0) @interpolate(flat, \(sampling)) value: u32) -> @location(0) u32 {
            return value;
        }
        """)
    let compiler = IndependentShaderCompiler()
    for entry in ["vs", "fs"] {
        if sampling == "first" {
            #expect(throws: ShaderFailure.unsupported("Render code generation: GLSL flat first sampling")) {
                try compiler.compileRenderStage(source, entryPoint: entry, to: .glsl)
            }
        } else {
            let artifact = try compiler.compileRenderStage(source, entryPoint: entry, to: .glsl)
            guard case .text(_, let code) = artifact.artifact else {
                Issue.record("Expected GLSL source"); continue
            }
            #expect(code.contains("flat \(entry == "vs" ? "out" : "in") uint"))
        }
    }
}

@Test func glslAcceptsDefaultFlatSamplingAndNonInterpolatedStageInterfaces() throws {
    let source = ShaderInput.wgsl("""
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(flat) value: u32,
        }
        @vertex fn vs(@location(0) input: u32) -> VertexOutput {
            return VertexOutput(vec4f(0.0), input);
        }
        @fragment fn fs(@location(0) @interpolate(flat) value: u32) -> @location(0) u32 {
            return value;
        }
        """)
    let compiler = IndependentShaderCompiler()
    for entry in ["vs", "fs"] {
        let artifact = try compiler.compileRenderStage(source, entryPoint: entry, to: .glsl)
        guard case .text(_, let code) = artifact.artifact else {
            Issue.record("Expected GLSL source"); continue
        }
        #expect(code.contains("flat \(entry == "vs" ? "out" : "in") uint"))
        #expect(code.contains(entry == "vs" ? "in uint saga_attribute_0" : "out uint saga_color"))
    }
}
