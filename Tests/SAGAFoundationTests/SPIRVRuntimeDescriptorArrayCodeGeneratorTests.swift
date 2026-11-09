import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func runtimeDescriptorArrayInput(varying: Bool) -> ShaderInput {
    let parameter = varying ? "@builtin(local_invocation_index) lane: u32" : ""
    let index = varying ? "lane" : "0u"
    let sampler = varying ? "1u - lane" : "1u"
    return .wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
        @group(0) @binding(1) var states: binding_array<sampler>;
        @group(0) @binding(2) var targets: binding_array<texture_storage_2d<rgba8unorm, write>>;
        @compute @workgroup_size(\(varying ? 2 : 1)) fn run(\(parameter)) {
            let color = textureSampleLevel(images[\(index)], states[\(sampler)], vec2f(1.25, 0.5), 0.0);
            textureStore(targets[\(index)], vec2i(0), color);
        }
        """)
}

private let runtimeDescriptorArrayLayout: [IRResourceBinding: UInt32] = [
    .init(group: 0, binding: 0): 2,
    .init(group: 0, binding: 1): 2,
    .init(group: 0, binding: 2): 2,
]

private func exportRuntimeDescriptorArray(_ compiler: IndependentShaderCompiler, varying: Bool = true) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(runtimeDescriptorArrayInput(varying: varying), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func runtimeDescriptorInstructions(_ words: [UInt32]) throws -> [(opcode: UInt32, range: Range<Int>)] {
    try #require(words.count > 5)
    var result: [(UInt32, Range<Int>)] = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        result.append((words[offset] & 0xffff, offset..<(offset + count)))
        offset += count
    }
    return result
}

private func descriptorIndexingExtension(_ words: [UInt32], in instructions: [(opcode: UInt32, range: Range<Int>)]) -> Range<Int>? {
    instructions.first { instruction in
        guard instruction.opcode == 10 else { return false } // OpExtension
        let bytes = words[instruction.range.dropFirst()].flatMap { word in
            (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) }
        }
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self) == "SPV_EXT_descriptor_indexing"
    }?.range
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvRuntimeDescriptorArrayCodeGeneratorPreservesUnboundedArraysUntilTextLayout(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportRuntimeDescriptorArray(compiler)
    let instructions = try runtimeDescriptorInstructions(words)
    #expect(instructions.filter { $0.opcode == 29 }.count == 3) // OpTypeRuntimeArray
    let capabilities = Set(instructions.filter { $0.opcode == 17 && $0.range.count == 2 }.map { words[$0.range.lowerBound + 1] })
    #expect(capabilities.isSuperset(of: [5301, 5302, 5307, 5309]))
    #expect(descriptorIndexingExtension(words, in: instructions) != nil)
    let input = ShaderInput.spirv(words)
    let imported = try compiler.parseAndValidate(input, capabilities: .allKnown).module
    for binding in runtimeDescriptorArrayLayout.keys {
        let variable = try #require(imported.globalVariables.values.first { $0.binding == binding })
        guard case .bindingArray(_, .dynamic) = try imported.types.value(for: variable.type).kind else {
            Issue.record("Expected an unbounded descriptor array at binding \(binding.binding)")
            continue
        }
    }
    do {
        _ = try compiler.compileCompute(input, to: format)
        Issue.record("Compiled a runtime descriptor array without its layout")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .invalidBinding })
    }
    let shader = try compiler.compileCompute(input, to: format, bindingArrayLengths: runtimeDescriptorArrayLayout)
    #expect(shader.sampledTextures.map { $0.bindingArray?.count } == [2])
    #expect(shader.samplerBindings.map { $0.bindingArray?.count } == [2])
    #expect(shader.storageTextures.map { $0.bindingArray?.count } == [2])
    #expect(shader.storageTextures.map(\.access) == [.store])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .allKnown)
    }
}

@Test func spirvRuntimeDescriptorArrayCodeGeneratorRequiresCapabilityAndExtension() throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportRuntimeDescriptorArray(compiler, varying: false)
    let instructions = try runtimeDescriptorInstructions(words)
    let capability = try #require(instructions.first {
        $0.opcode == 17 && $0.range.count == 2 && words[$0.range.lowerBound + 1] == 5302
    })
    let extensionRange = try #require(descriptorIndexingExtension(words, in: instructions))
    for range in [capability.range, extensionRange] {
        var malformed = words
        malformed.removeSubrange(range)
        do {
            _ = try compiler.parseAndValidate(.spirv(malformed), capabilities: .allKnown)
            Issue.record("Accepted a runtime descriptor array missing its capability or extension")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.contains { $0.code == .unsupportedCapability })
        }
    }
}

#if canImport(Metal)
@Test func spirvRuntimeDescriptorArrayCodeGeneratorUsesSuppliedLayoutOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(try exportRuntimeDescriptorArray(compiler))
    let shader = try compiler.compileCompute(input, to: .msl, bindingArrayLengths: runtimeDescriptorArrayLayout)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let image = TextureDescriptor(width: 2, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination])
        let target = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.storageBinding, .copySource])
        let first = try await device.createTexture(image)
        let second = try await device.createTexture(image)
        let output0 = try await device.createTexture(target)
        let output1 = try await device.createTexture(target)
        let clamp = try await device.createSampler(.init())
        let repeatU = try await device.createSampler(.init(addressModeU: .repeat))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let region = TextureCopyRegion(width: 2, height: 1)
        let snapshots = try await device.submitCommands([
            .writeTexture(destination: first, data: [255, 0, 0, 255, 0, 255, 0, 255], region: region),
            .writeTexture(destination: second, data: [0, 0, 255, 255, 255, 255, 255, 255], region: region),
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1), textures: [
                .init(binding: 0, arrayElement: 0): first, .init(binding: 0, arrayElement: 1): second,
                .init(binding: 2, arrayElement: 0): output0, .init(binding: 2, arrayElement: 1): output1,
            ], samplers: [.init(binding: 1, arrayElement: 0): clamp, .init(binding: 1, arrayElement: 1): repeatU]),
            .copyTextureToBuffer(source: output0, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: output1, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(snapshots == [[0xff0000ff], [0xffffffff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
