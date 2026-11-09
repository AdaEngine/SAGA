import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvPointerInput = ShaderInput.wgsl("""
    struct Packet { values: vec2u, stamp: u32 }
    @group(0) @binding(0) var<storage, read_write> published: u32;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 2>;
    var<workgroup> sharedValue: u32;
    var<private> invocationState: u32 = 5u;

    fn mutate(scalar: ptr<function, u32>, packet: ptr<function, Packet>, state: ptr<private, u32>, amount: u32) -> u32 {
        let previous = (*packet).values.y + *scalar + *state;
        *scalar += amount;
        (*packet).stamp += *scalar;
        (*packet).values.x += amount;
        *state += amount;
        return previous;
    }
    fn relay(scalar: ptr<function, u32>, packet: ptr<function, Packet>, state: ptr<private, u32>, amount: u32) -> u32 {
        return mutate(&*scalar, &*packet, &*state, amount);
    }
    fn publish(destination: ptr<storage, u32, read_write>, shared: ptr<workgroup, u32>, value: u32, groupValue: u32) {
        *destination = value;
        *shared = groupValue;
    }
    fn relayGlobals(destination: ptr<storage, u32, read_write>, shared: ptr<workgroup, u32>, value: u32, groupValue: u32) {
        publish(destination, shared, value, groupValue);
    }
    fn readShared(shared: ptr<workgroup, u32>) -> u32 { return *shared; }

    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
        var scalar = lane + 10u;
        var packet = Packet(vec2u(20u, 30u), 40u);
        let previous = relay(&scalar, &packet, &invocationState, lane + 1u);
        if lane == 0u {
            relayGlobals(&published, &sharedValue, scalar + packet.stamp, packet.values.x);
        }
        storageBarrier();
        workgroupBarrier();
        output[lane] = vec4u(previous, scalar, packet.stamp, published + readShared(&sharedValue) + invocationState);
    }
    """)

private func exportPointers(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvPointerInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvPointerCodeGeneratorRoundTripsFunctionPointersAndSpecializedGlobals(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportPointers(compiler)
    try #require(words.count > 5)
    var pointerSpaces: [UInt32: UInt32] = [:]
    var pointerParameters: [UInt32] = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 32 where count == 4: pointerSpaces[words[offset + 1]] = words[offset + 2] // OpTypePointer
        case 55 where count == 3:
            if let space = pointerSpaces[words[offset + 1]] { pointerParameters.append(space) }
        default: break
        }
        offset += count
    }
    #expect(pointerParameters.sorted() == [6, 6, 7, 7, 7, 7]) // Private and Function in both helpers.
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [4, 32])
    #expect(shader.workgroupStorageByteSize == 16)
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

#if canImport(Metal)
@Test func spirvPointerCodeGeneratorPreservesForwardedPointerWritesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportPointers(compiler)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let published = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): published, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
            .copy(source: published, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(snapshots == [[45, 11, 51, 89, 46, 13, 53, 90], [62]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
