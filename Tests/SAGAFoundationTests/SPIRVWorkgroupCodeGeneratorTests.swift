import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvWorkgroupInput = ShaderInput.wgsl("""
    var<workgroup> scratch: array<u32, 4>;
    var<workgroup> signedCount: atomic<i32>;
    var<workgroup> unsignedCount: atomic<u32>;
    @group(0) @binding(0) var<storage, read_write> mailbox: array<u32, 8>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32, 64>;

    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32,
                                     @builtin(workgroup_id) group: vec3u) {
        let initial = scratch[lane] | bitcast<u32>(atomicLoad(&signedCount)) | atomicLoad(&unsignedCount);
        workgroupBarrier();
        scratch[lane] = group.x * 100u + lane + 10u;
        var signedOld = 0i;
        var unsignedOld = 0u;
        if lane == 0u {
            atomicExchange(&signedCount, -5i);
            signedOld = atomicAdd(&signedCount, 2i);
            atomicStore(&unsignedCount, 4u);
            unsignedOld = atomicAdd(&unsignedCount, 3u);
        }
        workgroupBarrier();

        let nextLane = (lane + 1u) % 4u;
        let neighbor = scratch[nextLane];
        let uniformValue = workgroupUniformLoad(&scratch[2u]);
        let signedFinal = workgroupUniformLoad(&signedCount);
        let unsignedFinal = workgroupUniformLoad(&unsignedCount);

        let groupBase = group.x * 4u;
        mailbox[groupBase + lane] = group.x * 1000u + lane + 50u;
        storageBarrier();
        let forwarded = mailbox[groupBase + nextLane];
        let base = (groupBase + lane) * 8u;
        output[base] = initial;
        output[base + 1u] = neighbor;
        output[base + 2u] = uniformValue;
        output[base + 3u] = bitcast<u32>(signedOld);
        output[base + 4u] = unsignedOld;
        output[base + 5u] = bitcast<u32>(signedFinal);
        output[base + 6u] = unsignedFinal;
        output[base + 7u] = forwarded;
    }
    """)

private func exportWorkgroup(_ compiler: IndependentShaderCompiler, input: ShaderInput = spirvWorkgroupInput) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvWorkgroupCodeGeneratorRoundTripsSharedMemoryAndBarriers(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportWorkgroup(compiler)
    try #require(words.count > 5)
    var constants: [UInt32: UInt32] = [:]
    var opcodes: Set<UInt32> = []
    var semantics: [UInt32] = []
    var workgroupVariables = 0
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        switch opcode {
        case 43 where count == 4: constants[words[offset + 2]] = words[offset + 3]
        case 59 where count >= 4 && words[offset + 3] == 4: workgroupVariables += 1
        case 224 where count == 4:
            #expect(constants[words[offset + 1]] == 2 && constants[words[offset + 2]] == 2)
            semantics.append(try #require(constants[words[offset + 3]]))
        default: break
        }
        offset += count
    }
    #expect(workgroupVariables == 3)
    #expect(opcodes.isSuperset(of: [224, 227, 228, 229, 234]))
    #expect(semantics.contains { $0 & 0x108 == 0x108 })
    #expect(semantics.contains { $0 & 0x48 == 0x48 })
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.workgroupSize == .init(x: 4))
    #expect(shader.workgroupStorageByteSize == 48)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [32, 256])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

@Test(arguments: [false, true])
func spirvWorkgroupCodeGeneratorInitializesNestedAtomicsWithOneLocalIndexInput(_ structured: Bool) throws {
    let declaration = structured ? "struct Inputs { @builtin(local_invocation_index) index: u32 }" : ""
    let parameter = structured ? "input: Inputs" : "@builtin(local_invocation_id) id: vec3u"
    let lane = structured ? "input.index" : "id.x"
    let input = ShaderInput.wgsl("""
        \(declaration)
        struct Shared { counters: array<atomic<u32>, 2>, stamp: u32 }
        var<workgroup> sharedData: Shared;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 2>;
        @compute @workgroup_size(2) fn run(\(parameter)) {
            let lane = \(lane);
            output[lane] = atomicLoad(&sharedData.counters[lane]) + sharedData.stamp;
        }
        """)
    let compiler = IndependentShaderCompiler()
    let words = try exportWorkgroup(compiler, input: input)
    try #require(words.count > 5)
    var localIndices: [UInt32] = []
    var interfaces: Set<UInt32> = []
    var storesAtomic = false
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 15 where count >= 4: interfaces.formUnion(words[(offset + 4)..<(offset + count)]) // Entry point name "run" occupies one word.
        case 71 where count == 4 && words[offset + 2] == 11 && words[offset + 3] == 29:
            localIndices.append(words[offset + 1]) // BuiltIn LocalInvocationIndex
        case 228: storesAtomic = true
        default: break
        }
        offset += count
    }
    #expect(localIndices.count == 1)
    #expect(localIndices.allSatisfy { interfaces.contains($0) })
    #expect(storesAtomic)
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    #expect(shader.workgroupStorageByteSize == 16)
}

#if canImport(Metal)
@Test func spirvWorkgroupCodeGeneratorPreservesCollectivesAndZeroesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportWorkgroup(compiler)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let mailbox = try await device.createBuffer(.init(size: 32, usage: .storage))
        let output = try await device.createBuffer(.init(size: 256, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 256, usage: [.copyDestination, .mapRead]))
        let dispatch: [BufferCommand] = [
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): mailbox, .init(binding: 1): output],
                             workgroups: .init(x: 2)),
            .copy(source: output, destination: readback, byteCount: 256), .read(buffer: readback, byteCount: 256),
        ]
        let snapshots = try await device.submitCommands(dispatch + dispatch).wait()
        var expected: [UInt32] = []
        for group: UInt32 in 0..<2 {
            for lane: UInt32 in 0..<4 {
                let next = (lane + 1) % 4
                expected += [0, group * 100 + next + 10, group * 100 + 12,
                             lane == 0 ? 0xfffffffb : 0, lane == 0 ? 4 : 0,
                             0xfffffffd, 7, group * 1000 + next + 50]
            }
        }
        #expect(snapshots == [expected, expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
