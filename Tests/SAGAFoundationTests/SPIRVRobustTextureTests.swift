import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvRobustTextureInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read> arguments: array<vec4i, 6>;
    @group(0) @binding(1) var sampled: texture_2d_array<f32>;
    @group(0) @binding(2) var image: texture_storage_2d_array<r32uint, read_write>;
    @group(0) @binding(3) var<storage, read_write> output: array<vec4u, 12>;

    @compute @workgroup_size(6) fn run(@builtin(local_invocation_index) lane: u32) {
        let args = arguments[lane];
        let sampledValue = textureLoad(sampled, args.xy, args.z, args.w);
        let previous = textureLoad(image, args.xy, args.z);
        textureStore(image, args.xy, args.z, vec4u(previous.x + 100u + lane, 0u, 0u, 1u));
        output[lane * 2u] = bitcast<vec4u>(sampledValue);
        output[lane * 2u + 1u] = previous;
    }
    """)

private func exportRobustTexture(_ compiler: IndependentShaderCompiler, input: ShaderInput = spirvRobustTextureInput) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func expectGuardedTexelOperations(_ words: [UInt32], operations: Set<UInt32>, queries: Set<UInt32>) throws {
    try #require(words.count > 5)
    var checkedBlocks: Set<UInt32> = []
    var texelOperations: [(opcode: UInt32, block: UInt32)] = []
    var opcodes: Set<UInt32> = []
    var block: UInt32 = 0
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        switch opcode {
        case 248 where count == 2: block = words[offset + 1] // OpLabel
        case 250 where count >= 4: checkedBlocks.insert(words[offset + 2]) // True branch target.
        case 95, 98, 99: texelOperations.append((opcode, block)) // ImageFetch, ImageRead, ImageWrite
        default: break
        }
        offset += count
    }
    #expect(Set(texelOperations.map(\.opcode)) == operations)
    #expect(texelOperations.allSatisfy { checkedBlocks.contains($0.block) })
    #expect(opcodes.isSuperset(of: queries))
}

@Test func spirvRobustTextureGuardsTexelOperationsInTheBinary() throws {
    let words = try exportRobustTexture(IndependentShaderCompiler())
    try expectGuardedTexelOperations(words, operations: [95, 98, 99], queries: [103, 104, 106])
}

@Test(arguments: [false, true])
func spirvRobustTextureGuardsMultisampleFetchUsingSampleCount(extraStorageCapability: Bool) throws {
    let input = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read> arguments: vec4i;
        @group(0) @binding(1) var image: texture_multisampled_2d<f32>;
        @group(0) @binding(2) var<storage, read_write> output: vec4f;
        @compute @workgroup_size(1) fn run() {
            output = textureLoad(image, arguments.xy, arguments.z);
        }
        """)
    let compiler = IndependentShaderCompiler()
    var words = try exportRobustTexture(compiler, input: input)
    if extraStorageCapability {
        words.insert(contentsOf: SPIRVFrontendTests.i(17, 27), at: 5) // StorageImageMultisample
    }
    try expectGuardedTexelOperations(words, operations: [95], queries: [104, 107]) // ImageQuerySize, ImageQuerySamples
    let shader = try compiler.compileCompute(.spirv(words), to: .wgsl)
    #expect(shader.sampledTextures.map(\.isMultisampled) == [true])
    guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.wgsl(text))
}

extension SPIRVFrontendTests {
    @Test func spirvStorageImageMultisampleCapabilityDoesNotAdmitUnsupportedImageTypes() throws {
        let words = Self.head + Self.i(17, 1) + Self.i(17, 27) + Self.i(14, 0, 1)
            + Self.i(22, 1, 32) + Self.i(25, 2, 1, 1, 0, 0, 1, 2, 1) // multisampled Rgba32f storage image
        do {
            _ = try parse(words)
            Issue.record("Accepted an unsupported multisampled storage image")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.first?.code == .unsupportedConstruct)
            #expect(diagnostics.first?.message.contains("single-sampled storage images") == true)
        }
    }
}

#if canImport(Metal)
@Test func spirvRobustTexturePreservesArrayTexelBoundsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportRobustTexture(compiler)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let arguments = try await device.createBuffer(.init(size: 96, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 192, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 192, usage: [.copyDestination, .mapRead]))
        let sampled = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: [.textureBinding, .copyDestination], mipLevelCount: 2, arrayLayerCount: 2))
        let storage = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint,
            usage: [.storageBinding, .copyDestination, .copySource], arrayLayerCount: 2))
        let indices: [UInt32] = [1, 0, 1, 1, 2, 0, 1, 1, 0, 0, 2, 0,
                                 0, 0, 0, 2, 0, 0, .max, .max, .max, 0, 1, 0]
        let snapshots = try await device.submitCommands([
            .write(buffer: arguments, offset: 0, words: indices),
            .writeTexture(destination: sampled, data: [0, 255, 0, 255], region: .init(x: 1, width: 1, height: 1, mipLevel: 1, arrayLayer: 1)),
            .writeTexture(destination: storage, data: [10, 0, 0, 0, 11, 0, 0, 0], region: .init(width: 2, height: 1)),
            .writeTexture(destination: storage, data: [20, 0, 0, 0, 21, 0, 0, 0], region: .init(width: 2, height: 1, arrayLayer: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): arguments, .init(binding: 3): output],
                             workgroups: .init(x: 1), textures: [.init(binding: 1): sampled, .init(binding: 2): storage]),
            .copy(source: output, destination: readback, byteCount: 192), .read(buffer: readback, byteCount: 192),
            .copyTextureToBuffer(source: storage, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
            .copyTextureToBuffer(source: storage, destination: readback, bytesPerRow: 256, arrayLayer: 1),
            .read(buffer: readback, byteCount: 8),
        ]).wait()
        let zero: [UInt32] = [0, 0, 0, 0]
        let green: [UInt32] = [0, Float(1).bitPattern, 0, Float(1).bitPattern]
        let expected = [green, [21, 0, 0, 1], zero, zero, zero, zero, zero, [10, 0, 0, 1], zero, zero, zero, zero].flatMap { $0 }
        #expect(snapshots == [expected, [113, 11], [20, 121]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
