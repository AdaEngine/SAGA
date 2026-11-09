import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvImageAtomicInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var unsignedImage: texture_storage_2d<r32uint, atomic>;
    @group(0) @binding(1) var signedImage: texture_storage_2d<r32sint, atomic>;
    @group(0) @binding(2) var<storage, read> coordinates: array<vec2i, 5>;

    @compute @workgroup_size(1) fn run() {
        let mix = coordinates[0];
        let minimum = coordinates[1];
        let maximum = coordinates[2];
        let negative = coordinates[3];
        let high = coordinates[4];
        textureAtomicAdd(unsignedImage, mix, 3u);
        textureAtomicAnd(unsignedImage, mix, 7u);
        textureAtomicOr(unsignedImage, mix, 8u);
        textureAtomicXor(unsignedImage, mix, 1u);
        textureAtomicMin(unsignedImage, minimum, 5u);
        textureAtomicMax(unsignedImage, maximum, 9u);
        textureAtomicAdd(signedImage, mix, -5i);
        textureAtomicAnd(signedImage, mix, -2i);
        textureAtomicOr(signedImage, mix, 1i);
        textureAtomicXor(signedImage, mix, 1i);
        textureAtomicMin(signedImage, minimum, -5i);
        textureAtomicMax(signedImage, maximum, 3i);
        textureAtomicAdd(unsignedImage, negative, 100u);
        textureAtomicAdd(unsignedImage, high, 200u);
        textureAtomicAdd(signedImage, negative, 300i);
        textureAtomicAdd(signedImage, high, 400i);
    }
    """)

private func exportImageAtomics(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvImageAtomicInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvImageAtomicCodeGeneratorRoundTripsGuardedTexelPointers(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportImageAtomics(compiler)
    try #require(words.count > 5)
    var pointerSpaces: [UInt32: UInt32] = [:]
    var checkedBlocks: Set<UInt32> = []
    var texelPointers: [UInt32: (block: UInt32, type: UInt32)] = [:]
    var atomicPointers: [UInt32] = []
    var opcodes: Set<UInt32> = []
    var block: UInt32 = 0
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        switch opcode {
        case 32 where count == 4: pointerSpaces[words[offset + 1]] = words[offset + 2]
        case 60 where count == 6: texelPointers[words[offset + 2]] = (block, words[offset + 1]) // OpImageTexelPointer
        case 234, 236...242:
            if count >= 4 { atomicPointers.append(words[offset + 3]) }
        case 248 where count == 2: block = words[offset + 1]
        case 250 where count >= 4: checkedBlocks.insert(words[offset + 2])
        default: break
        }
        offset += count
    }
    #expect(atomicPointers.count == 16)
    #expect(atomicPointers.allSatisfy { id in
        guard let pointer = texelPointers[id] else { return false }
        return pointerSpaces[pointer.type] == 11 && checkedBlocks.contains(pointer.block) // Image storage class.
    })
    #expect(opcodes.isSuperset(of: [104, 234, 236, 237, 238, 239, 240, 241, 242]))
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.storageTextures.map(\.format) == [.r32Uint, .r32Sint])
    #expect(shader.storageTextures.allSatisfy { $0.access == [.load, .store, .atomic] })
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.textureAtomic))
    }
}

#if canImport(Metal)
@Test func spirvImageAtomicCodeGeneratorPreservesSignedExtremaAndBoundsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportImageAtomics(compiler)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.textureAtomic) else {
        await device.close()
        return
    }
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let usage: TextureUsage = [.storageAtomic, .copyDestination, .copySource]
        let unsignedImage = try await device.createTexture(.init(width: 4, height: 1, format: .r32Uint, usage: usage))
        let signedImage = try await device.createTexture(.init(width: 4, height: 1, format: .r32Sint, usage: usage))
        let coordinates = try await device.createBuffer(.init(size: 40, usage: [.storage, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let region = TextureCopyRegion(width: 4, height: 1)
        func bytes(_ words: [UInt32]) -> [UInt8] {
            words.flatMap { word in (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) } }
        }
        let snapshots = try await device.submitCommands([
            .writeTexture(destination: unsignedImage, data: bytes([10, 3, 6, 2]), region: region),
            .writeTexture(destination: signedImage, data: bytes([0xfffffffb, 0xfffffffe, 2, 0xfffffffe]), region: region),
            .write(buffer: coordinates, offset: 0, words: [1, 0, 2, 0, 3, 0, .max, 0, 4, 0]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): coordinates], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): unsignedImage, .init(binding: 1): signedImage]),
            .copyTextureToBuffer(source: unsignedImage, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 16),
            .copyTextureToBuffer(source: signedImage, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [[10, 15, 5, 9], [0xfffffffb, 0xfffffff8, 0xfffffffb, 3]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
