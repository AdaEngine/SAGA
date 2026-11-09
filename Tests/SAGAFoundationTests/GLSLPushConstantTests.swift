import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let pushConstantInput = ShaderInput.glsl("""
    #version 450
    layout(local_size_x = 1) in;
    layout(push_constant) uniform Parameters {
        uvec2 axis;
        layout(offset = 16) uvec2 pair;
        uint last;
    };
    layout(set = 0, binding = 0) buffer Output { uint words[3]; } outputData;
    void main() {
        outputData.words[0] = axis.y;
        outputData.words[1] = pair.x;
        outputData.words[2] = last;
    }
    """, stage: .compute, defines: [:])

private func pushConstantSPIRV(_ compiler: IndependentShaderCompiler) throws -> ShaderInput {
    guard case .spirv(let words) = try compiler.compile(pushConstantInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return .spirv(words)
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslPushConstantsPreserveImmediateOffsetsAndReflectNoDescriptor(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for input in [pushConstantInput, try pushConstantSPIRV(compiler)] {
        let shader = try compiler.compileCompute(input, to: format)
        #expect(shader.immediateLayout?.byteSize == 32)
        #expect(shader.uniformBindings.isEmpty)
        #expect(shader.storageBindings.map(\.minimumByteSize) == [12])
        if format == .wgsl, case .text(.wgsl, let source) = shader.artifact {
            _ = try compiler.parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.immediates))
        }
    }
}

@Test(arguments: [
    "layout(push_constant, binding=0) uniform Params { uint value; } data; void main() {}",
    "layout(push_constant) buffer Params { uint value; } data; void main() {}",
    "layout(push_constant) uniform Params { uint value; } data; void main() { data.value = 1u; }",
])
func glslPushConstantsRejectDescriptorsAndWrites(_ source: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileCompute(.glsl("#version 450\n" + source, stage: .compute, defines: [:]), to: .msl)
    }
}

#if canImport(Metal)
@Test func glslPushConstantsRoundTripAndReadExactMetalWords() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(pushConstantSPIRV(compiler), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 12, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                             immediates: [1, 2, 3, 4, 7, 8, 9, 10]),
            .copy(source: output, destination: readback, byteCount: 12), .read(buffer: readback, byteCount: 12),
        ]).wait()
        #expect(result == [[2, 7, 9]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
