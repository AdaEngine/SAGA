import SAGACompiler
import SAGAShaderIR
import Testing

private let float64ToInt32Input = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 2>;

    fn run(values: vec2<f64>) {
        output[0] = vec4u(bitcast<vec2u>(vec2i(values)), vec2u(values));
        output[1] = vec4u(bitcast<u32>(i32(values.x)), u32(values.y), 0u, 0u);
    }
    @compute @workgroup_size(1) fn main() { run(vec2(1., 2.)); }
    """)

private let float64ToInt64Input = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read_write> output: array<vec2<u64>, 2>;

    fn run(values: vec2<f64>) {
        output[0] = vec2<u64>(vec2<i64>(values));
        output[1] = vec2<u64>(u64(values.x), u64(i64(values.y)));
    }
    @compute @workgroup_size(1) fn main() { run(vec2(1., 2.)); }
    """)

@Test(arguments: [ShaderOutputFormat.hlsl, .glsl])
func wgslFloat64IntegerConversionUsesDoublePrecisionBounds(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let suffix = format == .hlsl ? "L" : "lf"
    let cases: [(ShaderInput, [Double])] = [
        (float64ToInt32Input, [Double(Int32.min), Double(Int32.max), Double(UInt32.max)]),
        (float64ToInt64Input, [Double(Int64.min), Double(Int64.max).nextDown, Double(UInt64.max).nextDown]),
    ]
    for (input, bounds) in cases {
        let shader = try compiler.compileCompute(input, to: format)
        guard case .text(_, let source) = shader.artifact else {
            throw ShaderFailure.incompatibleArtifact
        }
        for bound in bounds { #expect(source.contains("\(bound)\(suffix)")) }
        let bitcast = format == .hlsl ? "asuint(" : "unpackDouble2x32("
        #expect(source.contains(bitcast))
        #expect(source.contains("0x7ff00000u"))
    }
}

@Test func wgslFloat64IntegerConversionSPIRVGuardsScalarAndVectorCasts() throws {
    let compiler = IndependentShaderCompiler()
    let cases: [(ShaderInput, Bool, [Double])] = [
        (float64ToInt32Input, false, [Double(Int32.min), Double(Int32.max), Double(UInt32.max)]),
        (float64ToInt64Input, true, [Double(Int64.min), Double(Int64.max).nextDown, Double(UInt64.max).nextDown]),
    ]
    for (input, requiresInt64, bounds) in cases {
        guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
            throw ShaderFailure.incompatibleArtifact
        }
        // The importer expands ordered double comparisons into 64-bit integer keys.
        let module = try compiler.parseAndValidate(.spirv(words), capabilities: [.float64, .shaderInt64]).module
        for bound in bounds {
            #expect(module.globalExpressions.values.contains { $0.kind == .literal(.f64(bound)) })
        }
        var offset = 5
        var capabilities = Set<UInt32>()
        var guardedInputs = Set<UInt32>()
        var conversions: [[UInt32]] = []
        while offset < words.count {
            let count = Int(words[offset] >> 16)
            try #require(count > 0 && count <= words.count - offset)
            let opcode = words[offset] & 0xffff
            switch opcode {
            case 17: capabilities.insert(words[offset + 1]) // OpCapability
            case 169: guardedInputs.insert(words[offset + 2]) // OpSelect result ID
            case 109, 110: conversions.append([opcode, words[offset + 3]]) // OpConvertFToU / OpConvertFToS
            default: break
            }
            offset += count
        }
        #expect(capabilities.contains(10)) // Float64
        #expect(capabilities.contains(11) == requiresInt64) // Int64
        #expect(Set(conversions.map { $0[0] }) == [109, 110])
        #expect(conversions.allSatisfy { guardedInputs.contains($0[1]) })
    }
}
