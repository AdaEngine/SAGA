import SAGACompiler
import SAGAShaderIR
import Testing

@Test func wgslShortCircuitGuardsFunctionBodyConstantEvaluation() throws {
    let source = ShaderInput.wgsl("""
        const DISABLED = false;
        @group(0) @binding(0) var<storage, read_write> result: u32;
        @compute @workgroup_size(1) fn main() {
            let skippedRemainder = DISABLED && (9.5f % 0.0f > 1.0f);
            let skippedRoot = (3 < 4) || sqrt(-2.0f) > 1.0f;
            result = u32(skippedRemainder) + u32(skippedRoot);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileWebGPUCompute(source).artifact
    guard case .text(.wgsl, let output) = artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.wgsl(output))
    _ = try compiler.compile(source, to: .msl)
}

@Test(arguments: [
    "false && (3u + 2.0f > 1.0f)",
    "false && missing",
    "false && (1.0f +)",
    "false && (1i / 0i > 0i)",
    "false && (vec2u(1u)[2] == 0u)",
    "false && (f16(1) > 0)",
    "true && (9.5f % 0.0f > 1.0f)",
])
func wgslShortCircuitStillChecksTypesAndMandatoryErrors(_ expression: String) throws {
    let input = ShaderInput.wgsl("fn test() -> bool { return \(expression); }")
    #expect(throws: ShaderFailure.self) { _ = try IndependentShaderCompiler().parseAndValidate(input) }
}

@Test(arguments: [.wgsl, .msl, .hlsl, .glsl, .spirv] as [ShaderOutputFormat])
func wgslFloatRemainderSupportsScalarVectorAndMixedOperands(_ format: ShaderOutputFormat) throws {
    let input = ShaderInput.wgsl("""
        enable f16;
        const_assert (-11.5 % 4) == -3.5;
        const_assert (11.5f % -4) == 3.5f;
        const_assert all((vec2h(-11.5h, 11.5h) % 4h) == vec2h(-3.5h, 3.5h));
        const_assert all((11.5f % vec2f(-4.0f, 5.0f)) == vec2f(3.5f, 1.5f));
        @group(0) @binding(0) var<storage, read_write> values: vec4f;
        @compute @workgroup_size(1) fn main() {
            var pair = values.xy;
            pair %= values.z;
            let halfRemainder = f16(values.x) % 4h;
            values = vec4f(pair, values.z % values.w, f32(halfRemainder));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compile(input, to: format)
    try artifact.validateContainer(for: format)
    switch artifact {
    case .text(.wgsl, let source):
        #expect(source.contains(" % "))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
    case .text(.glsl, let source):
        #expect(source.contains("trunc("))
    case .text(.msl, let source), .text(.hlsl, let source):
        #expect(source.contains("fmod("))
    case .spirv(let words):
        var offset = 5
        var remainders = 0
        while offset < words.count {
            let instruction = words[offset]
            let wordCount = Int(instruction >> 16)
            try #require(wordCount > 0 && wordCount <= words.count - offset)
            if instruction & 0xffff == 140 { remainders += 1 } // OpFRem
            offset += wordCount
        }
        #expect(remainders == 3)
    default: throw ShaderFailure.incompatibleArtifact
    }
}
