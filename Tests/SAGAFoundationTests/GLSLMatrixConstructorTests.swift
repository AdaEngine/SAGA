import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let glslMatrixConstructorInput = ShaderInput.glsl("""
    #version 450
    layout(local_size_x=1) in;
    layout(set=0,binding=0,std430) buffer Results { vec4 values[5]; } result;
    void main() {
        mat2 diagonal = mat2(3);
        mat2 mixed = mat2(ivec2(1, 2), 3u, true);
        mat3x4 expanded = mat3x4(mixed);
        mat2x3 cropped = mat2x3(expanded);
        mat2x2 partial = mat2x2(6, vec4(7, 8u, 9.0, 10));
        mat4 packedMatrix = mat4(vec2(2), vec4(1), vec2(2), vec4(1), vec4(1));
        result.values[0] = vec4(diagonal[0], diagonal[1]);
        result.values[1] = expanded[2];
        result.values[2] = vec4(cropped[1], 10.0);
        result.values[3] = vec4(partial[0], partial[1]);
        result.values[4] = packedMatrix[1];
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslMatrixConstructorsReachTextBackends(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(glslMatrixConstructorInput, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [80])
    if format == .wgsl {
        guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [
    "mat2()",
    "mat2(1.0, 2.0, 3.0)",
    "mat2(vec4(1.0), 2.0)",
    "mat2(mat2(1), 0.0)",
    "mat2(mat2(1), mat2(2))",
])
func glslMatrixConstructorsRejectInvalidArguments(expression: String) throws {
    let source = "#version 450\nvoid main() { mat2 result = \(expression); }"
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.glsl(source, stage: .fragment, defines: [:]))
        Issue.record("Accepted a malformed GLSL matrix constructor")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func glslMatrixConstructorsPreserveColumnOrderAndIdentityFillOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(glslMatrixConstructorInput, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let result = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): result], workgroups: .init(x: 1)),
            .copy(source: result, destination: readback, byteCount: 80),
            .read(buffer: readback, byteCount: 80),
        ]).wait()
        let expected: [Float] = [3, 0, 0, 3, 0, 0, 1, 0, 3, 1, 0, 10, 6, 7, 8, 9, 1, 1, 2, 2]
        #expect(words == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
