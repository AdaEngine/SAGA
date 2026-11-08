import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let adjacentTemplateShader = """
    const A = 1u;
    const B = 2u;
    const C = 3u;
    const D = 4u;
    const_assert A<B || C>D;
    const counted: array<u32, (8u >> 2u)>=array(4u, 9u);
    const nested: array<u32, array<u32, 2>(2u, 3u)[0]>=array(5u, 11u);
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;

    @compute @workgroup_size(array<u32, 1>(1u)[0])
    fn run() {
        var sized: array<u32, 1u << 1u>=array(13u, 17u);
        let pointer: ptr<function, array<u32, 2>>=&sized;
        let pair: vec2<u32>=vec2u(19, 23);
        let bits = bitcast<vec2<u32>>(vec2f(1.0, 2.0));
        var shifted = 8u;
        shifted <<= 1u;
        shifted >>= 3u;
        var compared: array<u32, select(1u, 2u, B>=A)>=array(29u, 31u);

        output[0] = counted[1] + nested[0];
        output[1] = pointer[1] + pair.x;
        output[2] = select(0u, compared[1] + shifted, shifted>=B);
        output[3] = bits.x;
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func wgslTemplateDiscoverySeparatesClosingAnglesFromAssignment(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(adjacentTemplateShader), to: format)
    #expect(shader.storageBindings.first?.minimumByteSize == 16)
    guard case .text(let actual, let source) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(source)) }
}

@Test func wgslTemplateDiscoveryPreservesSpansAfterSplittingGreaterEqual() throws {
    let source = "@compute @workgroup_size(1) fn main() { let café: vec2<u32>=; }"
    let offset = try #require(Array(source.utf8).firstIndex(of: 59))
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Accepted a missing initializer")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .unexpectedToken)
        #expect(issues.first?.span?.startUTF8 == offset)
        #expect(issues.first?.span?.endUTF8 == offset + 1)
    }
}

#if canImport(Metal)
@Test func wgslTemplateDiscoveryExecutesAdjacentAssignmentsAndShiftsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(adjacentTemplateShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [[14, 36, 33, 0x3f800000]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
