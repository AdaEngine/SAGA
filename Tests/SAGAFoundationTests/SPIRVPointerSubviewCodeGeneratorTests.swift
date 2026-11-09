import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal

@Test func spirvPointerSubviewCodeGeneratorPreservesNestedDerivedPointersOnMetal() async throws {
    let source = ShaderInput.wgsl("""
        requires unrestricted_pointer_parameters, pointer_composite_access;
        struct Packet { samples: array<u32, 2>, sentinel: u32 }
        struct Root { tag: u32, packet: Packet }
        var<private> state = Root(200u, Packet(array(50u, 60u), 222u));
        var<private> calls: u32;
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 4>;

        fn mutate(local: ptr<function, u32>, privateValue: ptr<private, u32>) -> vec2u {
            calls += 1u;
            let before = *local + *privateValue;
            *local += 3u;
            *privateValue += 7u;
            return vec2u(before, *local + *privateValue);
        }
        fn relay(localPacket: ptr<function, Packet>, privateRoot: ptr<private, Root>, index: i32, lane: u32) -> vec2u {
            return mutate(&(*localPacket).samples[index], &(*privateRoot).packet.samples[lane]);
        }
        fn forward(localRoot: ptr<function, Root>, privateRoot: ptr<private, Root>, index: i32, lane: u32) -> vec2u {
            return relay(&(*localRoot).packet, privateRoot, index, lane);
        }
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            var local = Root(100u, Packet(array(10u, 20u), 111u));
            let result = forward(&local, &state, i32(lane) - 1i, lane);
            output[lane] = vec4u(result,
                local.tag + local.packet.samples[0] + local.packet.samples[1] + local.packet.sentinel + 1000u * calls,
                state.tag + state.packet.samples[0] + state.packet.samples[1] + state.packet.sentinel);
        }
        """)
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    let input = ShaderInput.spirv(words)
    guard case .text(.wgsl, let text) = try compiler.compileCompute(input, to: .wgsl).artifact else {
        throw ShaderFailure.incompatibleArtifact
    }
    _ = try compiler.parseAndValidate(.wgsl(text))
    let shader = try compiler.compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        ]).wait()
        #expect(snapshots == [[50, 57, 1241, 539, 70, 80, 1244, 539, 20, 23, 1244, 532, 0, 0, 1241, 532]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
