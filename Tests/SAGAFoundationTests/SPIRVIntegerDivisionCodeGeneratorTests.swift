import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal

@Test func spirvIntegerDivisionCodeGeneratorPreservesMixedVectorEdgesOnMetal() async throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read> inputs: array<vec4u, 4>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;
        @compute @workgroup_size(1) fn run() {
            let signedValue = bitcast<vec4i>(inputs[0]);
            let signedDivisor = bitcast<vec4i>(inputs[1]);
            let unsignedValue = inputs[2];
            let unsignedDivisor = inputs[3];
            output[0] = bitcast<vec4u>(signedValue / signedDivisor);
            output[1] = bitcast<vec4u>(signedValue % signedDivisor);
            output[2] = unsignedValue / unsignedDivisor;
            output[3] = unsignedValue % unsignedDivisor;
        }
        """)
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let binary = ShaderInput.spirv(words)
    guard case .text(.wgsl, let text) = try compiler.compileCompute(binary, to: .wgsl).artifact else {
        throw ShaderFailure.incompatibleArtifact
    }
    let inputs: [UInt32] = [0x80000000, 0xfffffff9, 7, 0xffffffed, .max, 0, 0xfffffffd, 4,
                            .max, 9, 17, 0x80000000, 2, 0, 5, 3]
    let expected: [UInt32] = [0x80000000, 0xfffffff9, 0xfffffffe, 0xfffffffc, 0, 0, 1, 0xfffffffd,
                              0x7fffffff, 9, 3, 715827882, 1, 0, 2, 2]
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        for roundTrip in [binary, .wgsl(text)] {
            let shader = try compiler.compileCompute(roundTrip, to: .msl)
            let pipeline = try await device.createComputePipeline(shader)
            let snapshots = try await device.submitCommands([
                .write(buffer: input, offset: 0, words: inputs),
                .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                                 workgroups: .init(x: 1)),
                .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
            ]).wait()
            #expect(snapshots == [expected])
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
