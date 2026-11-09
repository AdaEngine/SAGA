import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvRobustDataInput = ShaderInput.wgsl("""
    struct Element { value: u32, counter: atomic<u32> }
    struct Store { fixed: array<vec2u, 2>, marker: u32, tail: array<Element> }
    @group(0) @binding(0) var<storage, read> indices: vec4u;
    @group(0) @binding(1) var<uniform> cfg: array<vec4u, 2>;
    @group(0) @binding(2) var<storage, read_write> data: Store;
    @group(0) @binding(3) var<storage, read_write> output: array<vec4u, 5>;
    var<workgroup> scratch: array<vec2u, 2>;

    @compute @workgroup_size(1) fn run() {
        let negative = bitcast<i32>(indices.x);
        let valid = indices.y;
        let high = indices.z;
        scratch[0] = vec2u(51u, 52u);
        scratch[1] = vec2u(61u, 62u);

        output[0] = vec4u(cfg[negative].x, cfg[valid].x, cfg[high].x, arrayLength(&data.tail));
        output[1] = vec4u(data.fixed[negative].y, data.fixed[valid].y, data.fixed[high].y, data.marker);
        output[2] = vec4u(scratch[negative].x, scratch[valid].x, scratch[high].x, data.tail[valid].value);
        let negativeAtomic = atomicAdd(&data.tail[negative].counter, 100u);
        let highAtomic = atomicAdd(&data.tail[high].counter, 200u);
        output[3] = vec4u(data.tail[negative].value, data.tail[high].value, negativeAtomic, highAtomic);
        let old = atomicAdd(&data.tail[valid].counter, 5u);

        data.fixed[negative].x = 901u;
        data.fixed[high].x = 902u;
        data.fixed[valid].x = 55u;
        scratch[negative].x = 903u;
        scratch[high].x = 904u;
        scratch[valid].x = 65u;
        data.tail[negative].value = 905u;
        data.tail[high].value = 906u;
        data.tail[valid].value = 75u;
        output[4] = vec4u(old, scratch[0].x, scratch[1].x, atomicLoad(&data.tail[valid].counter));
    }
    """)

private func exportRobustDataAccess(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvRobustDataInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test func spirvRobustDataAccessGuardsDynamicAccessChainsInTheBinary() throws {
    let words = try exportRobustDataAccess(IndependentShaderCompiler())
    try #require(words.count > 5)
    var pointerSpaces: [UInt32: UInt32] = [:]
    var constants: Set<UInt32> = []
    var checkedBlocks: Set<UInt32> = []
    var accesses: [(block: UInt32, space: UInt32)] = []
    var block: UInt32 = 0
    var afterInitialization = false
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 32 where count == 4: pointerSpaces[words[offset + 1]] = words[offset + 2] // OpTypePointer
        case 43 where count >= 4: constants.insert(words[offset + 2])
        case 46 where count == 3: constants.insert(words[offset + 2])
        case 224: afterInitialization = true // The workgroup initializer uses its own bounded loop.
        case 248 where count == 2: block = words[offset + 1] // OpLabel
        case 250 where count >= 4: checkedBlocks.insert(words[offset + 2]) // True target of OpBranchConditional.
        case 65, 66:
            if afterInitialization, count >= 5, let space = pointerSpaces[words[offset + 1]], [2, 4, 12].contains(space),
               words[(offset + 4)..<(offset + count)].contains(where: { !constants.contains($0) }) {
                accesses.append((block, space))
            }
        default: break
        }
        offset += count
    }
    #expect(Set(accesses.map(\.space)) == [2, 4, 12]) // Uniform, Workgroup, StorageBuffer
    #expect(accesses.allSatisfy { checkedBlocks.contains($0.block) })
}

#if canImport(Metal)
@Test func spirvRobustDataAccessReadsZeroAndSkipsWritesOutsideBoundsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportRobustDataAccess(compiler)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let indices = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let config = try await device.createBuffer(.init(size: 32, usage: [.uniform, .copyDestination]))
        let data = try await device.createBuffer(.init(size: 36, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: indices, offset: 0, words: [.max, 1, .max, 0]),
            .write(buffer: config, offset: 0, words: [101, 102, 103, 104, 201, 202, 203, 204]),
            .write(buffer: data, offset: 0, words: [11, 12, 21, 22, 99, 31, 3, 41, 7]),
            .dispatchCompute(pipeline: pipeline, bindings: [
                .init(binding: 0): indices, .init(binding: 1): config, .init(binding: 2): data, .init(binding: 3): output,
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 80), .read(buffer: readback, byteCount: 80),
            .copy(source: data, destination: readback, byteCount: 36), .read(buffer: readback, byteCount: 36),
        ]).wait()
        let expected: [UInt32] = [0, 201, 0, 2, 0, 22, 0, 99, 0, 61, 0, 41, 0, 0, 0, 0, 7, 51, 65, 12]
        #expect(snapshots == [expected, [11, 12, 55, 22, 99, 31, 3, 75, 12]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
