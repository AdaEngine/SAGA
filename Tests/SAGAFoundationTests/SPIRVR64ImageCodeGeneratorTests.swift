import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvR64ImageAtomicInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var image: texture_storage_2d<r64uint, atomic>;

    @compute @workgroup_size(1) fn run() {
        textureAtomicMax(image, vec2i(0, 0), 0xfedcba9876543210lu);
        textureAtomicMin(image, vec2i(1, 0), 9lu);
        textureAtomicMin(image, vec2i(2, 0), 0x9000000000000001lu);
        textureAtomicMax(image, vec2i(3, 0), 11lu);
        textureAtomicMax(image, vec2i(-1, 0), 18446744073709551615lu);
        textureAtomicMin(image, vec2i(4, 0), 0lu);
    }
    """)

private let spirvR64ImageCopyInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var input: texture_storage_2d<r64uint, read>;
    @group(0) @binding(1) var output: texture_storage_2d<r64uint, write>;

    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let pixel = vec2u(lane, 0u);
        textureStore(output, pixel, textureLoad(input, pixel));
    }
    """)

private func exportR64Image(_ input: ShaderInput, with compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [false, true])
func spirvR64ImageCodeGeneratorDeclaresImageCapabilityAndGuardsAtomics(_ atomic: Bool) throws {
    let input = atomic ? spirvR64ImageAtomicInput : spirvR64ImageCopyInput
    let words = try exportR64Image(input, with: IndependentShaderCompiler())
    var capabilities: Set<UInt32> = []
    var extensions: Set<String> = []
    var wideTypes: Set<UInt32> = []
    var sampledTypes: [UInt32] = []
    var formats: Set<UInt32> = []
    var checkedBlocks: Set<UInt32> = []
    var texelBlocks: [UInt32: UInt32] = [:]
    var atomicOperands: [UInt32] = []
    var opcodes: Set<UInt32> = []
    var block: UInt32 = 0
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        switch opcode {
        case 10: // OpExtension
            let bytes = words[(offset + 1)..<(offset + count)].flatMap { word in
                (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) }
            }
            extensions.insert(String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self))
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 21 where count == 4 && words[offset + 2] == 64 && words[offset + 3] == 0:
            wideTypes.insert(words[offset + 1])
        case 25 where count >= 9: // OpTypeImage
            sampledTypes.append(words[offset + 2])
            formats.insert(words[offset + 8])
        case 60 where count == 6: texelBlocks[words[offset + 2]] = block
        case 237 where count == 7, 239 where count == 7: atomicOperands.append(words[offset + 3]) // UMin, UMax
        case 248 where count == 2: block = words[offset + 1]
        case 250 where count >= 4: checkedBlocks.insert(words[offset + 2])
        default: break
        }
        offset += count
    }
    #expect(extensions.contains("SPV_EXT_shader_image_int64"))
    #expect(capabilities.contains(5016)) // Int64ImageEXT
    #expect(capabilities.contains(12) == atomic) // Int64Atomics
    #expect(formats == [40]) // R64ui
    #expect(sampledTypes.allSatisfy { wideTypes.contains($0) })
    #expect(opcodes.contains(104)) // OpImageQuerySize for robust texel bounds
    if atomic {
        #expect(opcodes.isSuperset(of: [237, 239]))
        #expect(atomicOperands.count == 6)
        #expect(atomicOperands.allSatisfy { texelBlocks[$0].map(checkedBlocks.contains) == true })
    } else {
        #expect(opcodes.isSuperset(of: [98, 99])) // OpImageRead, OpImageWrite
        #expect(texelBlocks.isEmpty)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvR64ImageCodeGeneratorRoundTripsImageFormats(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for (input, access): (ShaderInput, [IRStorageAccess]) in [
        (spirvR64ImageAtomicInput, [[.load, .store, .atomic]]), (spirvR64ImageCopyInput, [.load, .store]),
    ] {
        let shader = try compiler.compileCompute(.spirv(try exportR64Image(input, with: compiler)), to: format)
        #expect(shader.storageTextures.map(\.format) == Array(repeating: .r64Uint, count: access.count))
        #expect(shader.storageTextures.map(\.access) == access)
        if format == .hlsl { #expect(shader.hlslProfile == "cs_6_6") }
        if format == .wgsl, case .text(.wgsl, let text) = shader.artifact {
            _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union([.shaderInt64, .textureInt64Atomic]))
        }
    }
}

#if canImport(Metal)
@Test func spirvR64ImageCodeGeneratorPreservesUnsignedExtremaOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let atomic = try compiler.compileCompute(.spirv(try exportR64Image(spirvR64ImageAtomicInput, with: compiler)), to: .msl)
    let copy = try compiler.compileCompute(.spirv(try exportR64Image(spirvR64ImageCopyInput, with: compiler)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.textureInt64Atomic) else {
        await device.close()
        return
    }
    do {
        let atomicPipeline = try await device.createComputePipeline(atomic)
        let copyPipeline = try await device.createComputePipeline(copy)
        let image = try await device.createTexture(.init(width: 4, height: 1, format: .r64Uint,
            usage: [.storageAtomic, .storageBinding, .copyDestination]))
        let output = try await device.createTexture(.init(width: 4, height: 1, format: .r64Uint,
            usage: [.storageBinding, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
        let seeds: [UInt64] = [7, 0x8000_0000_0000_0001, .max, .max]
        let bytes = seeds.flatMap { withUnsafeBytes(of: $0.littleEndian) { Array($0) } }
        let words = try await device.submitCommands([
            .writeTexture(destination: image, data: bytes, region: .init(width: 4, height: 1)),
            .dispatchCompute(pipeline: atomicPipeline, bindings: [:], workgroups: .init(x: 1), textures: [.init(binding: 0): image]),
            .dispatchCompute(pipeline: copyPipeline, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): image, .init(binding: 1): output]),
            .copyTextureToBuffer(source: output, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 32),
        ]).wait()
        #expect(words == [[0x7654_3210, 0xfedc_ba98, 9, 0, 1, 0x9000_0000, .max, .max]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
