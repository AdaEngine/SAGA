import SAGACompiler
import SAGAShaderIR
import Testing

@Test(arguments: [
    ("1D", "vec2", "textureProj", ""),
    ("1D", "vec4", "textureProjGradOffset", ", 0.25, 0.5, 1"),
    ("2D", "vec3", "textureProjOffset", ", ivec2(1), 0.5"),
    ("2D", "vec4", "textureProjLod", ", 1.0"),
    ("3D", "vec4", "textureProjGrad", ", vec3(0.25), vec3(0.5)"),
    ("3D", "vec4", "textureProjLodOffset", ", 1.0, ivec3(1)"),
])
func glslProjectiveSamplesDivideSpatialCoordinatesByLastLane(
    _ dimension: String, _ packedType: String, _ builtin: String, _ extraOperands: String
) throws {
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform texture\(dimension) image;
        layout(binding=1) uniform sampler state;
        layout(location=0) out vec4 color;
        \(packedType) coordinates() { return \(packedType)(gl_FragCoord.x + 1.0); }
        void main() {
            color = \(builtin)(sampler\(dimension)(image, state), coordinates()\(extraOperands));
        }
        """, stage: .fragment, defines: [:])
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let calls = main.statements.values.filter {
        if case .call = $0.kind { return true }
        return false
    }
    #expect(calls.count == 1)
    let sample = try #require(main.expressions.values.compactMap { expression -> IRImageSample? in
        if case .imageSample(let sample) = expression.kind { return sample }
        return nil
    }.first)
    let coordinate = try main.expressions.value(for: sample.coordinate)
    guard case .binary(.divide, let left, let right) = coordinate.kind else {
        Issue.record("Expected a division of spatial coordinates")
        return
    }
    let rightExpression = try main.expressions.value(for: right)
    let divisor = if case .splat(_, let value) = rightExpression.kind { try main.expressions.value(for: value) }
        else { rightExpression }
    guard case .accessIndex(let packed, let lastIndex) = divisor.kind else {
        Issue.record("Expected the last component of the projective coordinate")
        return
    }
    let packedValue = try main.expressions.value(for: packed)
    #expect(lastIndex == (packedType == "vec2" ? 1 : packedType == "vec3" ? 2 : 3))
    guard case .callResult = packedValue.kind else {
        Issue.record("Expected the projective coordinate to be evaluated by one call")
        return
    }
    let numerator = try main.expressions.value(for: left)
    if dimension == "1D" {
        #expect(numerator.kind == .accessIndex(base: packed, index: 0))
    } else {
        let width: IRVectorSize = dimension == "2D" ? .two : .three
        #expect(numerator.kind == .swizzle(size: width, vector: packed, pattern: .init(x: .x, y: .y, z: .z, w: .w)))
    }
    #expect((sample.offset != nil) == builtin.hasSuffix("Offset"))
    if builtin.contains("Grad") {
        guard case .gradient(let x, let y) = sample.level else {
            Issue.record("Expected explicit gradients")
            return
        }
        let dx = try main.expressions.value(for: x).kind
        let dy = try main.expressions.value(for: y).kind
        if dimension == "1D" {
            #expect(dx == .literal(.f32(0.25)))
            #expect(dy == .literal(.f32(0.5)))
        } else {
            guard case .splat(.three, let dxValue) = dx, case .splat(.three, let dyValue) = dy else {
                Issue.record("Expected original gradient vectors")
                return
            }
            #expect(try main.expressions.value(for: dxValue).kind == .literal(.f32(0.25)))
            #expect(try main.expressions.value(for: dyValue).kind == .literal(.f32(0.5)))
        }
    }
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    _ = try compiler.parseAndValidate(.spirv(words))
}

@Test(arguments: [("1D", "vec3"), ("2D", "vec2"), ("3D", "vec3"), ("2DArray", "vec4"), ("Cube", "vec4")])
func glslProjectiveSamplesRejectInvalidCoordinateAndTextureDimensions(_ dimension: String, _ packedType: String) {
    let input = ShaderInput.glsl("""
        #version 450
        layout(binding=0) uniform texture\(dimension) image;
        layout(binding=1) uniform sampler state;
        layout(location=0) out vec4 color;
        void main() {
            color = textureProj(sampler\(dimension)(image, state), \(packedType)(1.0));
        }
        """, stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}
