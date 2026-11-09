import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func storageImageArrayInput(varying: Bool) -> ShaderInput {
    let input = varying ? "@builtin(local_invocation_index) index: u32" : "@builtin(workgroup_id) group: vec3u"
    let setup = varying ? "" : "let index = group.x;"
    return .wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var outputs: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
        @compute @workgroup_size(\(varying ? 2 : 1)) fn storeArray(\(input)) {
            \(setup)
            let green = f32(index);
            textureStore(outputs[index], vec2i(0), vec4f(1.0 - green, green, 0.0, 1.0));
        }
        """)
}

private let depthImageArrayInput = ShaderInput.wgsl("""
    enable wgpu_binding_array;
    @group(0) @binding(0) var depths: binding_array<texture_depth_2d, 2>;
    @group(0) @binding(1) var comparisons: binding_array<sampler_comparison, 2>;
    @group(0) @binding(2) var<storage, read_write> output: array<vec2f, 2>;
    @compute @workgroup_size(2) fn compareArray(@builtin(local_invocation_index) lane: u32) {
        let uv = vec2f(0.5);
        output[lane] = vec2f(textureSampleCompareLevel(depths[lane], comparisons[1u - lane], uv, 0.5),
                            textureSampleCompareLevel(depths[lane], comparisons[1u - lane], uv, 0.125));
    }
    """)

private func exportImageArray(_ input: ShaderInput, with compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func descriptorArrayInstructions(_ words: [UInt32]) throws -> (capabilities: Set<UInt32>, nonUniform: Set<UInt32>, nonReadable: Set<UInt32>, opcodes: Set<UInt32>) {
    try #require(words.count > 5)
    var capabilities: Set<UInt32> = []
    var nonUniform: Set<UInt32> = []
    var nonReadable: Set<UInt32> = []
    var opcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        if opcode == 17 && count == 2 { capabilities.insert(words[offset + 1]) }
        if opcode == 71 && count == 3 {
            if words[offset + 2] == 25 { nonReadable.insert(words[offset + 1]) }
            if words[offset + 2] == 5300 { nonUniform.insert(words[offset + 1]) }
        }
        offset += count
    }
    return (capabilities, nonUniform, nonReadable, opcodes)
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvStorageDepthArrayCodeGeneratorRoundTripsStorageAccess(_ format: ShaderOutputFormat, _ varying: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportImageArray(storageImageArrayInput(varying: varying), with: compiler)
    let instructions = try descriptorArrayInstructions(words)
    #expect(instructions.capabilities.contains(31) == !varying) // StorageImageArrayDynamicIndexing
    #expect(instructions.capabilities.contains(5301) == varying)
    #expect(instructions.capabilities.contains(5309) == varying) // StorageImageArrayNonUniformIndexing
    #expect(instructions.nonUniform.isEmpty == !varying)
    #expect(instructions.nonReadable.count == 1)
    #expect(instructions.opcodes.contains(99)) // OpImageWrite
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "storeArray", to: format)
    let image = try #require(shader.storageTextures.first)
    #expect(image.bindingArray?.count == 2 && image.access == .store && image.format == .rgba8Unorm)
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.storageTextureBindingArray, .storageTextureBindingArrayNonUniformIndexing])
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvStorageDepthArrayCodeGeneratorRoundTripsDepthComparison(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportImageArray(depthImageArrayInput, with: compiler)
    let instructions = try descriptorArrayInstructions(words)
    #expect(instructions.capabilities.isSuperset(of: [5301, 5307]))
    #expect(!instructions.nonUniform.isEmpty)
    #expect(instructions.opcodes.contains(90)) // OpImageSampleDrefExplicitLod
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "compareArray", to: format)
    let image = try #require(shader.sampledTextures.first)
    let sampler = try #require(shader.samplerBindings.first)
    #expect(image.bindingArray?.count == 2 && image.isDepth)
    #expect(sampler.bindingArray?.count == 2 && sampler.isComparison)
    if format == .glsl {
        #expect(shader.glslTextureSamplerBindings.contains {
            $0.texture == image.binding && $0.sampler == sampler.binding && $0.isComparison
        })
    }
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.textureAndSamplerBindingArray, .textureAndSamplerBindingArrayNonUniformIndexing])
    }
}

@Test func spirvStorageDepthArrayCodeGeneratorRejectsMixedSamplerArrayRoles() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var depths: binding_array<texture_depth_2d, 2>;
        @group(0) @binding(1) var comparisons: binding_array<sampler_comparison, 2>;
        @group(0) @binding(2) var colors: binding_array<texture_2d<f32>, 2>;
        @group(0) @binding(3) var regular: binding_array<sampler, 2>;
        @group(0) @binding(4) var<storage, read_write> output: vec2f;
        @compute @workgroup_size(1) fn run() {
            let uv = vec2f(0.5);
            output = vec2f(textureSampleLevel(colors[0], regular[0], uv, 0.0).x,
                           textureSampleCompareLevel(depths[0], comparisons[0], uv, 0.5));
        }
        """)
    var words = try exportImageArray(source, with: compiler)
    var bindings: [UInt32: UInt32] = [:]
    var accessBases: [Int] = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 71 where count == 4 && words[offset + 2] == 33: bindings[words[offset + 3]] = words[offset + 1]
        case 65 where count >= 4: accessBases.append(offset + 3)
        default: break
        }
        offset += count
    }
    // A raw SPIR-V sampler has no comparison type. Redirect the Dref access to the ordinary array.
    let comparison = try #require(bindings[1])
    let regular = try #require(bindings[3])
    let comparisonAccess = try #require(accessBases.first { words[$0] == comparison })
    words[comparisonAccess] = regular
    do {
        _ = try compiler.compileCompute(.spirv(words), to: .msl)
        Issue.record("Accepted conflicting sampler roles on one descriptor array")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .unsupportedConstruct && $0.message.contains("regular and comparison") })
    }
}

#if canImport(Metal)
@Test func spirvStorageDepthArrayCodeGeneratorWritesEachStorageElementOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportImageArray(storageImageArrayInput(varying: true), with: compiler)
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "storeArray", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let descriptor = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.storageBinding, .copySource])
        let first = try await device.createTexture(descriptor)
        let second = try await device.createTexture(descriptor)
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0, arrayElement: 0): first, .init(binding: 0, arrayElement: 1): second]),
            .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(snapshots == [[0xff0000ff], [0xff00ff00]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func spirvStorageDepthArrayCodeGeneratorUsesEachDepthAndComparisonSamplerOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportImageArray(depthImageArrayInput, with: compiler)
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "compareArray", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let clearShader = ShaderInput.wgsl("""
            @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 0, 0, 1); }
            """)
        let render = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(clearShader, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(clearShader, entryPoint: "fs", to: .msl),
            colorFormat: .rgba8Unorm, depthState: .init(compare: .always))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
        let descriptor = TextureDescriptor(width: 1, height: 1, format: .depth32Float, usage: [.textureBinding, .renderAttachment])
        let first = try await device.createTexture(descriptor)
        let second = try await device.createTexture(descriptor)
        let greater = try await device.createSampler(.init(compare: .greater))
        let less = try await device.createSampler(.init(compare: .less))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        let snapshots = try await device.submitCommands([
            .render(pipeline: render, target: target, clear: clear, vertexCount: 0, depth: .init(texture: first, loadAction: .clear(0.75))),
            .render(pipeline: render, target: target, clear: clear, vertexCount: 0, depth: .init(texture: second, loadAction: .clear(0.25))),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0, arrayElement: 0): first, .init(binding: 0, arrayElement: 1): second],
                             samplers: [.init(binding: 1, arrayElement: 0): greater, .init(binding: 1, arrayElement: 1): less]),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        let expected: [Float] = [1, 1, 1, 0]
        #expect(snapshots == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
