import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let abstractVectorUnarySource = """
        @group(0) @binding(0) var<storage, read_write> result: array<vec2u, 4>;
        @compute @workgroup_size(1) fn main() {
            result[0] = bitcast<vec2u>(vec2i(-(vec2(1, 2))));
            result[1] = bitcast<vec2u>(vec2i(~(vec2(1, 2))));
            result[2] = bitcast<vec2u>(vec2f(-(vec2(1.5, 2.5))));
            result[3] = select(vec2u(0), vec2u(1), !vec2<bool>(true, false));
        }
        """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func abstractVectorUnaryOperations(format: ShaderOutputFormat) throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(abstractVectorUnarySource), to: format)
    guard case .text(_, let text) = shader.artifact else {
        Issue.record("Expected shader source")
        return
    }
    #expect(text.contains("4294967293u")) // Bit pattern of i32(-3).
    #expect(text.contains("3217031168u")) // Bit pattern of f32(-1.5).
}

@Test(arguments: ["~vec2(1.5, 2.5)", "!vec2(1, 2)", "!vec2(1.5, 2.5)"])
func rejectsInvalidAbstractVectorUnaryOperations(expression: String) {
    let source = "@compute @workgroup_size(1) fn main() { let value = \(expression); }"
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
    }
}

#if canImport(Metal)
@Test func abstractVectorUnaryMetalReadback() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(abstractVectorUnarySource), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 32),
            .read(buffer: readback, byteCount: 32),
        ])
        let words = try #require(try await ticket.wait().first)
        #expect(words == [
            UInt32(bitPattern: -1), UInt32(bitPattern: -2), UInt32(bitPattern: -2), UInt32(bitPattern: -3),
            Float(-1.5).bitPattern, Float(-2.5).bitPattern, 0, 1,
        ])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
