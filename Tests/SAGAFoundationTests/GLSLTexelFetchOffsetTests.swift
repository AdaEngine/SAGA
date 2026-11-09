import SAGACompiler
import SAGAShaderIR
import Testing

@Test(arguments: [
    ("1D", "texture", "int(gl_FragCoord.x)", "-2"),
    ("2D", "utexture", "ivec2(int(gl_FragCoord.x), 0)", "ivec2(-2, 1)"),
    ("2DArray", "itexture", "ivec3(int(gl_FragCoord.x), 0, 2)", "ivec2(-2, 1)"),
    ("3D", "texture", "ivec3(int(gl_FragCoord.x), 0, 0)", "ivec3(-2, 1, 3)"),
])
func glslTexelFetchOffsetAddsBeforeLoadAndPreservesLayer(
    _ dimension: String, _ resourceType: String, _ coordinate: String, _ offset: String
) throws {
    let samplerPrefix = resourceType == "texture" ? "" : String(resourceType.prefix(1))
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform \(resourceType)\(dimension) image;
        layout(binding=1) uniform sampler state;
        layout(location=0) out \(samplerPrefix)vec4 color;
        void main() {
            color = texelFetchOffset(\(samplerPrefix)sampler\(dimension)(image, state), \(coordinate), 1, \(offset));
        }
        """, stage: .fragment, defines: [:])
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let load = try #require(main.expressions.values.compactMap { expression -> IRImageLoad? in
        if case .imageLoad(let load) = expression.kind { return load }
        return nil
    }.first)
    #expect(load.sample == nil)
    let level = try #require(load.level)
    #expect(try main.expressions.value(for: level).kind == .literal(.i32(1)))
    guard case .binary(.add, let original, let shift) = try main.expressions.value(for: load.coordinate).kind else {
        Issue.record("Expected the offset to be applied before the image load")
        return
    }
    let shiftExpression = try main.expressions.value(for: shift).kind
    if dimension == "1D" {
        #expect(shiftExpression == .literal(.i32(-2)))
    } else {
        guard case .compose(_, let components) = shiftExpression else {
            Issue.record("Expected a spatial offset vector")
            return
        }
        let offsets = try components.map { try main.expressions.value(for: $0).kind }
        #expect(offsets == (dimension == "3D" ? [.literal(.i32(-2)), .literal(.i32(1)), .literal(.i32(3))]
            : [.literal(.i32(-2)), .literal(.i32(1))]))
    }
    if dimension == "2DArray" {
        let layer = try #require(load.arrayIndex)
        guard case .swizzle(.two, let packed, _) = try main.expressions.value(for: original).kind else {
            Issue.record("Expected the original spatial coordinate from the packed array coordinate")
            return
        }
        #expect(try main.expressions.value(for: layer).kind == .accessIndex(base: packed, index: 2))
    } else {
        #expect(load.arrayIndex == nil)
    }
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    _ = try compiler.parseAndValidate(.spirv(words))
    let wgsl = try compiler.compileWebGPURenderStage(input, entryPoint: "main")
    guard case .text(.wgsl, let output) = wgsl.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(output.contains("textureLoad("))
    _ = try compiler.parseAndValidate(.wgsl(output))
}

@Test(arguments: [
    ("2D", "ivec2(0)", "ivec2(8)"),
    ("2D", "ivec2(0)", "ivec2(int(gl_FragCoord.x), 0)"),
    ("2DArray", "ivec3(0)", "ivec3(0)"),
    ("2DMS", "ivec2(0)", "ivec2(0)"),
])
func glslTexelFetchOffsetRejectsUnsupportedOperands(_ dimension: String, _ coordinate: String, _ offset: String) {
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform texture\(dimension) image;
        layout(binding=1) uniform sampler state;
        layout(location=0) out vec4 color;
        void main() {
            color = texelFetchOffset(sampler\(dimension)(image, state), \(coordinate), 0, \(offset));
        }
        """, stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}
