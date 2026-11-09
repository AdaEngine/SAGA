import SAGACompiler
import SAGAFrontends
import SAGAShaderIR
import Testing

private let float64Input = ShaderInput.wgsl("""
    var<private> initial: f64 = 1.0000000000000002lf;
    const increment: f64 = 2.0lf;
    @group(0) @binding(0) var<storage, read_write> result: array<f32>;

    fn sum(value: f64) -> f64 {
        let local = 3e1lf + 4.0e2lf + f64(5);
        return value + local + increment - 1.0lf;
    }

    @compute @workgroup_size(1)
    fn main() {
        result[0] = f32(sum(initial));
    }
    """)

@Test func wgslNativeFloat64ParsesAndLowersWithoutNarrowing() throws {
    let compiler = IndependentShaderCompiler()
    let validated = try compiler.parseAndValidate(float64Input, capabilities: [.float64])
    #expect(validated.module.types.values.contains { $0.kind == .scalar(IRScalar(kind: .float, width: 8)) })
    for format: ShaderOutputFormat in [.hlsl, .glsl] {
        let shader = try compiler.compileCompute(float64Input, to: format)
        guard case .text(_, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains("double"))
    }
    guard case .spirv(let words) = try compiler.compile(float64Input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let imported = try compiler.parseAndValidate(.spirv(words), capabilities: [.float64])
    #expect(imported.module.types.values.contains { $0.kind == .scalar(IRScalar(kind: .float, width: 8)) })
    #expect(imported.module.globalExpressions.values.contains { $0.kind == .literal(.f64(1.0.nextUp)) })
}

@Test func wgslFloat64RequiresNativeCapabilityAndTarget() {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(float64Input) }
    for format: ShaderOutputFormat in [.msl, .wgsl] {
        #expect(throws: ShaderFailure.self) { try compiler.compileCompute(float64Input, to: format) }
    }
}

@Test func wgslFloat64VectorConstantInitializerPreservesDoubleLanes() throws {
    let shader = ShaderInput.wgsl("""
        var<private> pair: vec2<f64> = vec2<f64>(-1.5lf, 1.0000000000000002lf);
        @group(0) @binding(0) var<storage, read_write> result: f32;
        @compute @workgroup_size(1)
        fn main() { result = f32(pair.x + pair.y); }
        """)
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(shader, capabilities: [.float64]).module
    #expect(module.globalExpressions.values.contains { $0.kind == .literal(.f64(1.0.nextUp)) })
    let output = try compiler.compileCompute(shader, to: .hlsl)
    guard case .text(.hlsl, let source) = output.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("double2"))
}
