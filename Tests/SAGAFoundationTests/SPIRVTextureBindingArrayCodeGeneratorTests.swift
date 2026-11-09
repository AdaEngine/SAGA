import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvDescriptorCapabilities: ShaderCapabilities = [
    .textureAndSamplerBindingArray, .textureAndSamplerBindingArrayNonUniformIndexing,
]

private enum DescriptorIndexing: CaseIterable, Sendable {
    case constant, workgroupUniform, invocationVarying
}

private func textureBindingArrayExportInput(indexing: DescriptorIndexing) -> ShaderInput {
    let (input, setup, workgroupSize): (String, String, Int) = switch indexing {
    case .constant: ("", "", 1)
    case .workgroupUniform: ("@builtin(workgroup_id) group: vec3u", "let lane = group.x;", 1)
    case .invocationVarying: ("@builtin(local_invocation_index) lane: u32", "", 2)
    }
    let body = indexing == .constant ? """
        result[0] = bitcast<vec4u>(textureSampleLevel(images[0], states[1], uv, 0.0));
        result[1] = bitcast<vec4u>(textureSampleLevel(images[1], states[0], uv, 0.0));
        """ : """
        let samplerIndex = 1u - lane;
        result[lane] = bitcast<vec4u>(textureSampleLevel(images[lane], states[samplerIndex], uv, 0.0));
        """
    return .wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
        @group(0) @binding(1) var states: binding_array<sampler, 2>;
        @group(0) @binding(2) var<storage, read_write> result: array<vec4u, 2>;
        @compute @workgroup_size(\(workgroupSize)) fn sampleArrays(\(input)) {
            let uv = vec2f(1.25, 0.5);
            \(setup)
            \(body)
        }
        """)
}

private func exportTextureBindingArray(_ compiler: IndependentShaderCompiler, indexing: DescriptorIndexing) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(textureBindingArrayExportInput(indexing: indexing), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func expectSPIRVDescriptorIndexing(_ words: [UInt32], indexing: DescriptorIndexing) throws {
    try #require(words.count > 5)
    var capabilities: Set<UInt32> = []
    var extensions: Set<String> = []
    var nonUniform: Set<UInt32> = []
    var accesses: Set<UInt32> = []
    var sampledImages: [(result: UInt32, image: UInt32, sampler: UInt32)] = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let operands = Array(words[(offset + 1)..<(offset + count)])
        switch words[offset] & 0xffff {
        case 10: // OpExtension
            let bytes = operands.flatMap { word in (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) } }
            extensions.insert(String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self))
        case 17 where operands.count == 1: capabilities.insert(operands[0]) // OpCapability
        case 65 where operands.count >= 3: accesses.insert(operands[1]) // OpAccessChain
        case 71 where operands.count == 2 && operands[1] == 5300: nonUniform.insert(operands[0])
        case 86 where operands.count == 4: sampledImages.append((operands[1], operands[2], operands[3]))
        default: break
        }
        offset += count
    }
    let varying = indexing == .invocationVarying
    #expect(capabilities.contains(29) == (indexing == .workgroupUniform)) // SampledImageArrayDynamicIndexing
    #expect(extensions.contains("SPV_EXT_descriptor_indexing") == varying)
    #expect(capabilities.contains(5301) == varying) // ShaderNonUniform
    #expect(capabilities.contains(5307) == varying) // SampledImageArrayNonUniformIndexing
    if varying {
        #expect(accesses.intersection(nonUniform).count >= 2)
        let sample = try #require(sampledImages.first)
        #expect(nonUniform.contains(sample.result) || (nonUniform.contains(sample.image) && nonUniform.contains(sample.sampler)))
    } else {
        #expect(nonUniform.isEmpty)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], DescriptorIndexing.allCases)
private func spirvTextureBindingArrayCodeGeneratorRoundTripsFixedDescriptors(_ format: ShaderOutputFormat, _ indexing: DescriptorIndexing) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportTextureBindingArray(compiler, indexing: indexing)
    try expectSPIRVDescriptorIndexing(words, indexing: indexing)
    let input = ShaderInput.spirv(words)
    let module = try compiler.parseAndValidate(input, capabilities: spirvDescriptorCapabilities).module
    #expect(module.entryPoints.map(\.name) == ["sampleArrays"])
    let shader = try compiler.compileCompute(input, entryPoint: "sampleArrays", to: format)
    let image = try #require(shader.sampledTextures.first)
    let sampler = try #require(shader.samplerBindings.first)
    #expect(image.binding == .init(group: 0, binding: 0) && image.bindingArray?.count == 2)
    #expect(sampler.binding == .init(group: 0, binding: 1) && sampler.bindingArray?.count == 2)
    #expect(image.samplingSamplers == [sampler.binding])
    #expect(shader.storageBindings.map(\.minimumByteSize) == [32])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: spirvDescriptorCapabilities)
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvTextureBindingArrayCodeGeneratorSelectsTextureAndSamplerElementsOnMetal(_ varying: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportTextureBindingArray(compiler, indexing: varying ? .invocationVarying : .constant)
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "sampleArrays", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let descriptor = TextureDescriptor(width: 2, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination])
        let first = try await device.createTexture(descriptor)
        let second = try await device.createTexture(descriptor)
        let clamp = try await device.createSampler(.init())
        let repeatU = try await device.createSampler(.init(addressModeU: .repeat))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let region = TextureCopyRegion(width: 2, height: 1)
        let snapshots = try await device.submitCommands([
            .writeTexture(destination: first, data: [255, 0, 0, 255, 0, 255, 0, 255], region: region),
            .writeTexture(destination: second, data: [0, 0, 255, 255, 255, 255, 255, 255], region: region),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0, arrayElement: 0): first, .init(binding: 0, arrayElement: 1): second],
                             samplers: [.init(binding: 1, arrayElement: 0): clamp, .init(binding: 1, arrayElement: 1): repeatU]),
            .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
        ]).wait()
        let expected: [Float] = [1, 0, 0, 1, 1, 1, 1, 1]
        #expect(snapshots == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
