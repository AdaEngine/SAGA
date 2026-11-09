import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvLayoutArrayInput = ShaderInput.wgsl("""
    enable wgpu_binding_array;
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
    @group(0) @binding(1) var singleton: binding_array<texture_2d<f32>>;
    @group(0) @binding(2) var states: binding_array<sampler>;
    @group(0) @binding(3) var<storage, read_write> output: array<vec4u, 4>;
    @group(0) @binding(4) var targets: binding_array<texture_storage_2d<rgba8unorm, write>>;
    @group(0) @binding(5) var unused: binding_array<texture_2d<f32>>;

    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let index = i32(lane) - 1i;
        let color = textureSampleLevel(images[index], states[index], vec2f(1.25, 0.5), 0.0);
        output[lane] = vec4u(bitcast<vec3u>(color.rgb), textureDimensions(singleton[7]).x);
        if lane == 0u { textureStore(targets[7], vec2i(0), vec4f(1.0, 0.0, 1.0, 1.0)); }
    }
    """)

private let spirvArrayLayout: [IRResourceBinding: UInt32] = [
    .init(group: 0, binding: 0): 2,
    .init(group: 0, binding: 1): 1,
    .init(group: 0, binding: 2): 2,
    .init(group: 0, binding: 4): 2,
]

private func exportLayoutArray(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvLayoutArrayInput, to: .spirv, bindingArrayLengths: spirvArrayLayout) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvLayoutBindingArrayCodeGeneratorSpecializesCountsPerBinding(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let raw = try compiler.parseAndValidate(spirvLayoutArrayInput, capabilities: .allKnown).module
    let image = try #require(raw.globalVariables.values.first { $0.binding == .init(group: 0, binding: 0) })
    let singleton = try #require(raw.globalVariables.values.first { $0.binding == .init(group: 0, binding: 1) })
    #expect(image.type == singleton.type)

    let input = ShaderInput.spirv(try exportLayoutArray(compiler))
    let imported = try compiler.parseAndValidate(input, capabilities: .allKnown).module
    #expect(!imported.globalVariables.values.contains { $0.binding == .init(group: 0, binding: 5) })
    for (binding, count) in spirvArrayLayout {
        let global = try #require(imported.globalVariables.values.first { $0.binding == binding })
        guard case .bindingArray(_, .constant(let actual)) = try imported.types.value(for: global.type).kind else {
            Issue.record("SPIR-V descriptor array at binding \(binding.binding) did not have a fixed count")
            continue
        }
        #expect(actual == count)
    }
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.sampledTextures.map { $0.bindingArray?.count } == [2, 1])
    #expect(shader.samplerBindings.map { $0.bindingArray?.count } == [2])
    #expect(shader.storageTextures.map { $0.bindingArray?.count } == [2])
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .allKnown)
    }
}

@Test func spirvLayoutBindingArrayCodeGeneratorRejectsMissingZeroAndUnknownCounts() throws {
    let compiler = IndependentShaderCompiler()
    let images = IRResourceBinding(group: 0, binding: 0)
    let invalid = [
        spirvArrayLayout.filter { $0.key != images },
        spirvArrayLayout.merging([images: 0], uniquingKeysWith: { _, new in new }),
        spirvArrayLayout.merging([.init(group: 3, binding: 9): 2], uniquingKeysWith: { _, new in new }),
    ]
    for lengths in invalid {
        do {
            _ = try compiler.compile(spirvLayoutArrayInput, to: .spirv, bindingArrayLengths: lengths)
            Issue.record("Accepted an invalid SPIR-V descriptor-array layout")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.contains { $0.code == .invalidBinding })
        }
    }
}

#if canImport(Metal)
@Test func spirvLayoutBindingArrayCodeGeneratorClampsSignedAndStaticIndicesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportLayoutArray(compiler)
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let descriptor = TextureDescriptor(width: 2, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination])
        let first = try await device.createTexture(descriptor)
        let second = try await device.createTexture(descriptor)
        let singleton = try await device.createTexture(.init(width: 3, height: 1, format: .rgba8Unorm, usage: .textureBinding))
        let target = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.storageBinding, .copyDestination, .copySource])
        let target0 = try await device.createTexture(target)
        let target1 = try await device.createTexture(target)
        let clamp = try await device.createSampler(.init())
        let repeating = try await device.createSampler(.init(addressModeU: .repeat))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let region = TextureCopyRegion(width: 2, height: 1)
        let snapshots = try await device.submitCommands([
            .writeTexture(destination: first, data: [255, 0, 0, 255, 0, 255, 0, 255], region: region),
            .writeTexture(destination: second, data: [0, 0, 255, 255, 255, 255, 255, 255], region: region),
            .writeTexture(destination: target0, data: [0, 0, 0, 255], region: .init(width: 1, height: 1)),
            .writeTexture(destination: target1, data: [0, 0, 0, 255], region: .init(width: 1, height: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 3): output], workgroups: .init(x: 1), textures: [
                .init(binding: 0, arrayElement: 0): first, .init(binding: 0, arrayElement: 1): second,
                .init(binding: 1, arrayElement: 0): singleton,
                .init(binding: 4, arrayElement: 0): target0, .init(binding: 4, arrayElement: 1): target1,
            ], samplers: [.init(binding: 2, arrayElement: 0): clamp, .init(binding: 2, arrayElement: 1): repeating]),
            .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
            .copyTextureToBuffer(source: target0, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: target1, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]).wait()
        let green: [UInt32] = [0, Float(1).bitPattern, 0, 3]
        let blue: [UInt32] = [0, 0, Float(1).bitPattern, 3]
        #expect(snapshots == [green + green + blue + blue, [0xff000000], [0xffff00ff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
