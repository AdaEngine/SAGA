import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let glslVectorRelationalInput = ShaderInput.glsl("""
    #version 450
    precision highp float;
    precision mediump int;
    layout(local_size_x=1) in;
    layout(set=0,binding=0,std430) buffer Results { vec4 values[6]; } result;
    void main() {
        precision lowp float;
        ivec4 signedA = ivec4(-1, 2, 3, -4);
        ivec4 signedB = ivec4(0, 2, 1, -3);
        uvec4 unsignedB = uvec4(1u, 2u, 4u, 3u);
        vec4 left = vec4(1.0, 2.0, 3.0, 4.0);
        vec4 right = vec4(1.0, 0.0, 4.0, 3.0);
        bvec4 equalLanes = equal(left, right);
        result.values[0] = vec4(lessThan(signedA, signedB));
        result.values[1] = vec4(lessThan(signedA, unsignedB));
        result.values[2] = vec4(greaterThanEqual(left, right));
        result.values[3] = vec4(notEqual(equalLanes, bvec4(false, true, false, true)));
        result.values[4] = vec4(float(any(equalLanes)), float(all(equalLanes)),
                                float(not(equalLanes).y), float(lessThanEqual(left, right).z));
        result.values[5] = vec4(greaterThan(left, right));
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslVectorRelationsAndPrecisionReachTextBackends(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(glslVectorRelationalInput, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [96])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test func glslDoubleVectorRelationsRequireFloat64Capability() throws {
    let source = ShaderInput.glsl("""
        #version 450
        void main() {
            bvec2 result = lessThan(dvec2(1.0), vec2(2.0));
        }
        """, stage: .fragment, defines: [:])
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(source) }
    _ = try compiler.parseAndValidate(source, capabilities: [.float64])
}

@Test(arguments: [
    "bvec2 result = lessThan(1, 2);",
    "bvec2 result = lessThan(ivec2(0), ivec3(1));",
    "bvec2 result = lessThan(bvec2(false), bvec2(true));",
    "bvec2 result = equal(bvec2(false), ivec2(0));",
    "bool result = any(true);",
    "bool result = all(ivec2(0));",
    "bvec2 result = not(bvec2(false), bvec2(true));",
])
func glslVectorRelationsRejectInvalidOperandTypes(_ statement: String) throws {
    let source = ShaderInput.glsl("#version 450\nvoid main() { \(statement) }", stage: .fragment, defines: [:])
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source)
        Issue.record("Accepted an invalid GLSL vector relational operand")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .typeMismatch)
    }
}

@Test(arguments: ["precision extreme float;", "precision highp uint;", "precision highp float"])
func glslDefaultPrecisionRejectsMalformedDeclarations(_ declaration: String) {
    let source = ShaderInput.glsl("#version 450\n\(declaration)\nvoid main() {}", stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(source) }
}

#if canImport(Metal)
@Test func glslVectorRelationsUsePerLaneAndUnsignedComparisonOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(glslVectorRelationalInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let result = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 96, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): result], workgroups: .init(x: 1)),
            .copy(source: result, destination: readback, byteCount: 96),
            .read(buffer: readback, byteCount: 96),
        ]).wait()
        let expected: [Float] = [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
        #expect(words == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
