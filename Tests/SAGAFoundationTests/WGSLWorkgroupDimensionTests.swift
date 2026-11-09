import SAGACompiler
import SAGAShaderIR
import Testing

// The entry-point dimensions from Naga's pinned const-exprs.wgsl.
private let mixedWorkgroupSource = ShaderInput.wgsl("""
    const TWO: u32 = 2u;
    const THREE: i32 = 3i;
    @compute @workgroup_size(TWO, THREE, TWO - 1u) fn main() {}
    """)

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func nativeTranslationAcceptsNagaMixedWorkgroupDimensions(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(mixedWorkgroupSource, entryPoint: "main", to: format)
    #expect(shader.workgroupSize == .init(x: 2, y: 3, z: 1))
}

@Test func spirvNormalizesNagaMixedWorkgroupDimensionsForWebGPU() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(mixedWorkgroupSource, to: .spirv) else {
        Issue.record("Expected SPIR-V")
        return
    }
    let shader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "main")
    #expect(shader.workgroupSize == .init(x: 2, y: 3, z: 1))
}

@Test func webGPURejectsDifferentConcreteWorkgroupDimensionTypes() {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(mixedWorkgroupSource) }
    #expect(throws: ShaderFailure.self) { try compiler.compileCompute(mixedWorkgroupSource, to: .wgsl) }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileWebGPUCompute(mixedWorkgroupSource, entryPoint: "main")
    }
}

@Test(arguments: ["2i, 3, 1", "2, 3u, 1"])
func materializesAbstractWorkgroupDimensionsWithOneConcreteType(dimensions: String) throws {
    let source = ShaderInput.wgsl("@compute @workgroup_size(\(dimensions)) fn main() {}")
    let compiler = IndependentShaderCompiler()
    for shader in [
        try compiler.compileCompute(source, entryPoint: "main", to: .msl),
        try compiler.compileWebGPUCompute(source, entryPoint: "main"),
    ] {
        #expect(shader.workgroupSize == .init(x: 2, y: 3, z: 1))
    }
}

@Test func specializesMixedWorkgroupDimensionsForNamedNativeEntry() throws {
    let source = ShaderInput.wgsl("""
        @id(7) override width: u32 = 2u;
        override height: i32 = 3i;
        override unrelated: i32;
        @compute @workgroup_size(width + 1, height - 1, 2) fn main() {}
        @compute @workgroup_size(unrelated) fn other() {}
        """)
    let compiler = IndependentShaderCompiler()
    let defaults = try compiler.compileCompute(source, entryPoint: "main", to: .msl)
    let supplied = try compiler.compileCompute(
        source, entryPoint: "main", to: .msl, constants: ["7": 4, "height": 7])
    #expect(defaults.workgroupSize == .init(x: 3, y: 2, z: 2))
    #expect(supplied.workgroupSize == .init(x: 5, y: 6, z: 2))
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, entryPoint: "main", to: .msl, constants: ["height": 1])
    }
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(source, entryPoint: "main") }
}

@Test(arguments: ["2u + 1i", "2u, 0i", "2u, -1i", "2u, 1.0", "2u, 3i, 2147483648"])
func nativeWorkgroupDimensionsStillRequireValidIndividualExpressions(dimensions: String) {
    let source = ShaderInput.wgsl("@compute @workgroup_size(\(dimensions)) fn main() {}")
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileCompute(source, entryPoint: "main", to: .msl)
    }
}
