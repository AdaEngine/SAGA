import SAGACompiler
import SAGAShaderIR
import Testing

@Test func wgslFloatToIntegerCastRoundTripsInNestedVectorExpression() throws {
    let input = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> result: vec4f;

        fn converted(value: vec4f) -> vec4f {
            let signed = vec4i(vec4f(0.0f, 1.0f, 2.0f, 3.0f));
            return vec4f(vec4i(2) + signed) + value + value + value + value + value;
        }

        @compute @workgroup_size(1) fn main() {
            result = converted(result);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileWebGPUCompute(input).artifact
    guard case .text(.wgsl, let output) = artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.wgsl(output))
    #expect(!output.contains("clamp("))

    let metal = try compiler.compile(input, to: .msl)
    guard case .text(.msl, let output) = metal else { throw ShaderFailure.incompatibleArtifact }
    #expect(output.contains("clamp("))
}
