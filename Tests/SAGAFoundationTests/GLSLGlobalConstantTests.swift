import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

private let glslGlobalConstantSource = ShaderInput.glsl("""
    #version 450 core
    layout(local_size_x = 1) in;
    const uint count = 1 + 1u;
    const uint outputCount = count + 6;
    const float scale = 2;
    const vec4 source = vec4(scale, 3.0, -4.0, 7.0);
    const vec2 swizzled = source.zx + vec2(10.0, 3.0);
    const float[count] values = { swizzled.x, swizzled.y * scale };
    const vec2 pairs[count] = vec2[count](vec2(values[0], scale), swizzled);
    const bool ready = values[1] == 10;
    layout(std430, binding = 0) buffer Data { uint words[outputCount]; };

    void main() {
        uint index = words[0];
        words[1] = uint(values[index]);
        words[2] = uint(pairs[index].x);
        words[3] = uint(pairs[index].y);
        words[4] = uint(scale);
        words[5] = uint(ready);
        words[6] = count;
        words[7] = uint(values[0]);
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl, .spirv])
func glslGlobalTypedConstantsReachBackends(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(glslGlobalConstantSource).module
    #expect(module.constants.values.compactMap(\.name).contains("swizzled"))
    #expect(module.constants.values.compactMap(\.name).contains("pairs"))
    if format == .spirv {
        let result = try compiler.compile(glslGlobalConstantSource, to: .spirv)
        guard case .spirv(let words) = result else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.spirv(words))
    } else {
        let result = try compiler.compileCompute(glslGlobalConstantSource, to: format)
        #expect(result.storageBindings.map(\.minimumByteSize) == [32])
        if format == .wgsl, case .text(.wgsl, let source) = result.artifact {
            _ = try compiler.parseAndValidate(.wgsl(source))
        }
    }
}

@Test(arguments: [
    "const float bad = later; const float later = 1.0;",
    "float value; const float bad = value;",
    "float getValue() { return 1.0; } const float bad = getValue();",
    "const vec3 bad = vec2(1.0);",
    "const float[2] bad = { 1.0 };",
    "const float[2] bad = { 1.0, 2.0, 3.0 };",
    "const float[2] bad = float[3](1.0, 2.0, 3.0);",
    "const float bad = 1.0; const float bad = 2.0;",
])
func glslGlobalTypedConstantsRejectInvalidInitializers(_ declarations: String) {
    let input = ShaderInput.glsl("#version 450 core\n\(declarations)\nvoid main() {}", stage: .compute, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}

#if canImport(Metal)
@Test func glslGlobalTypedConstantsRunOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(glslGlobalConstantSource, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let words = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let results = try await device.submitCommands([
            .write(buffer: words, offset: 0, words: [1]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): words], workgroups: .init(x: 1)),
            .copy(source: words, destination: readback, byteCount: 32),
            .read(buffer: readback, byteCount: 32),
        ]).wait()
        #expect(results == [[1, 10, 6, 5, 2, 1, 2, 6]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
