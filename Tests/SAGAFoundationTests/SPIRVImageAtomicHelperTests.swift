import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func imageAtomicHelperModule(useDescriptorArray: Bool) -> [UInt32] {
    let i = SPIRVFrontendTests.i
    var words = SPIRVFrontendTests.head + i(17, 1) // Shader
    if useDescriptorArray {
        words += i(17, 5301) + i(17, 5309) // ShaderNonUniform, StorageImageArrayNonUniformIndexing
        words += i(10, 0x5f565053, 0x5f545845, 0x63736564, 0x74706972, 0x695f726f, 0x7865646e, 0x00676e69)
    }
    words += i(14, 0, 1)
    words += useDescriptorArray ? i(15, 5, 30, 0x006e7572, 22) : i(15, 5, 30, 0x006e7572)
    words += i(16, 30, 17, useDescriptorArray ? 2 : 1, 1, 1)
    words += i(71, 20, 34, 0) + i(71, 20, 33, 0) + i(71, 21, 34, 0) + i(71, 21, 33, 1)
    if useDescriptorArray {
        words += i(71, 22, 11, 29) // LocalInvocationIndex
        for id: UInt32 in [32, 33, 41, 45] { words += i(71, id, 5300) } // NonUniform
    }
    for instruction in [
        i(19, 1), i(21, 2, 32, 0), i(21, 3, 32, 1), i(23, 4, 3, 2),
        i(25, 5, 2, 1, 0, 0, 0, 2, 33), i(32, 6, 0, 5), i(32, 7, 11, 2),
        i(33, 8, 1), i(33, 9, 1, 6, 4, 2),
        i(43, 2, 10, 0), i(43, 2, 11, 1), i(43, 2, 12, 9),
        i(43, 3, 13, 0), i(43, 3, 14, 1), i(43, 3, 15, .max), i(43, 3, 16, 2),
        i(44, 4, 60, 13, 13), i(44, 4, 61, 14, 13), i(44, 4, 62, 15, 13), i(44, 4, 63, 16, 13),
        i(43, 2, 64, 5), i(43, 2, 65, 3), i(43, 2, 66, 100), i(43, 2, 67, 200),
    ] { words += instruction }
    if useDescriptorArray {
        words += i(43, 2, 17, 2) + i(28, 18, 5, 17) + i(32, 19, 0, 18) + i(32, 23, 1, 2)
    }
    words += i(59, 6, 20, 0) + i(59, useDescriptorArray ? 19 : 6, 21, 0)
    if useDescriptorArray { words += i(59, 23, 22, 1) }

    // The leaf takes a UniformConstant pointer, then builds an Image pointer for the atomics.
    words += i(54, 1, 40, 0, 9) + i(55, 6, 41) + i(55, 4, 42) + i(55, 2, 43) + i(248, 44)
    words += i(60, 7, 45, 41, 42, 10) + i(234, 2, 46, 45, 11, 10, 43) + i(239, 2, 47, 45, 11, 10, 12)
    words += i(253) + i(56)
    // The relay forwards the same resource pointer through a second call.
    words += i(54, 1, 50, 0, 9) + i(55, 6, 51) + i(55, 4, 52) + i(55, 2, 53) + i(248, 54)
    words += i(57, 1, 55, 40, 51, 52, 53) + i(253) + i(56)
    words += i(54, 1, 30, 0, 8) + i(248, 31)
    if useDescriptorArray { words += i(61, 2, 32, 22) + i(65, 6, 33, 21, 32) }
    let second: UInt32 = useDescriptorArray ? 33 : 21
    return words + i(57, 1, 34, 50, 20, 60, 64) + i(57, 1, 35, 50, second, 61, 65)
        + i(57, 1, 36, 50, 20, 62, 66) + i(57, 1, 37, 50, second, 63, 67) + i(253) + i(56)
}

private func exportImageAtomicHelpers(_ compiler: IndependentShaderCompiler, useDescriptorArray: Bool) throws -> [UInt32] {
    let input = ShaderInput.spirv(imageAtomicHelperModule(useDescriptorArray: useDescriptorArray))
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [false, true])
func spirvImageAtomicHelpersReExportValidDescriptorOperands(_ useDescriptorArray: Bool) throws {
    let words = try exportImageAtomicHelpers(IndependentShaderCompiler(), useDescriptorArray: useDescriptorArray)
    var pointerTypes: [UInt32: UInt32] = [:]
    var descriptorAddresses: [UInt32: UInt32] = [:]
    var texelAddresses: [UInt32: UInt32] = [:]
    var nonUniform: Set<UInt32> = []
    var capabilities: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 32 where count == 4: pointerTypes[words[offset + 1]] = words[offset + 2]
        case 59 where count >= 4, 65 where count >= 4: descriptorAddresses[words[offset + 2]] = words[offset + 1]
        case 60 where count == 6: texelAddresses[words[offset + 2]] = words[offset + 3]
        case 71 where count == 3 && words[offset + 2] == 5300: nonUniform.insert(words[offset + 1])
        default: break
        }
        offset += count
    }
    try #require(!texelAddresses.isEmpty)
    for address in texelAddresses.values {
        let pointerType = try #require(descriptorAddresses[address])
        #expect(pointerTypes[pointerType] == 0) // UniformConstant
    }
    #expect(capabilities.contains(5309) == useDescriptorArray)
    #expect(texelAddresses.keys.contains { nonUniform.contains($0) } == useDescriptorArray)
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvImageAtomicHelpersPreserveOrdinaryAndArrayBindings(_ format: ShaderOutputFormat, _ useDescriptorArray: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportImageAtomicHelpers(compiler, useDescriptorArray: useDescriptorArray)
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "run", to: format)
    try #require(shader.storageTextures.count == 2)
    #expect(shader.storageTextures.allSatisfy { $0.format == .r32Uint && $0.access == [.load, .store, .atomic] })
    #expect(shader.storageTextures[0].bindingArray == nil)
    #expect(shader.storageTextures[1].bindingArray?.count == (useDescriptorArray ? 2 : nil))
    guard case .text(_, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    switch format {
    case .msl: #expect(source.contains(".atomic_fetch_add("))
    case .hlsl: #expect(source.contains("InterlockedAdd("))
    case .glsl: #expect(source.contains("imageAtomicAdd("))
    case .wgsl:
        #expect(source.contains("textureAtomicAdd("))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union([
            .textureAtomic, .storageTextureBindingArray, .storageTextureBindingArrayNonUniformIndexing,
        ]))
    default: Issue.record("Unexpected text target")
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvImageAtomicHelpersUpdateDistinctBindingsOnMetal(_ useDescriptorArray: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportImageAtomicHelpers(compiler, useDescriptorArray: useDescriptorArray)
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "run", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.textureAtomic) else {
        await device.close()
        return
    }
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let usage: TextureUsage = [.storageAtomic, .copySource, .copyDestination]
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let region = TextureCopyRegion(width: 2, height: 1)
        let second = ComputeBindingSlot(binding: 1, arrayElement: useDescriptorArray ? 0 : nil)
        var seeds: [(ComputeBindingSlot, [UInt8])] = [(.init(binding: 0), [6, 17]), (second, [21, 2])]
        if useDescriptorArray { seeds.append((.init(binding: 1, arrayElement: 1), [40, 10])) }
        var textures: [ComputeBindingSlot: ResourceID] = [:]
        var uploads: [BufferCommand] = []
        var reads: [BufferCommand] = []
        for (slot, seed) in seeds {
            let texture = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint, usage: usage))
            textures[slot] = texture
            uploads.append(.writeTexture(destination: texture, data: seed.flatMap { [$0, 0, 0, 0] }, region: region))
            reads.append(.copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256))
            reads.append(.read(buffer: readback, byteCount: 8))
        }
        let dispatch = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1), textures: textures)
        let snapshots = try await device.submitCommands(uploads + [dispatch] + reads).wait()
        let expected: [[UInt32]] = useDescriptorArray ? [[16, 17], [21, 9], [40, 13]] : [[11, 17], [21, 9]]
        #expect(snapshots == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
