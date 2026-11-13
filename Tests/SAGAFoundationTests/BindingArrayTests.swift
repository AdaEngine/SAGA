import Testing
import SAGACompiler
import SAGAShaderIR
import SAGAValidation
import SAGATypes
import SAGACore
import SAGAMetal

private let textureArrayCapabilities: ShaderCapabilities = [.textureAndSamplerBindingArray, .storageTextureBindingArray]
private let bufferArrayCapabilities: ShaderCapabilities = [.bufferBindingArray, .storageBufferBindingArray]

@Test func wgslBindingArraysPreserveResourceTypesAndBounds() throws {
    let source = """
        enable wgpu_binding_array;
        alias Images = binding_array<texture_2d<f32>, 1 + 2>;
        @group(0) @binding(0) var images: Images;
        @group(0) @binding(1) var states: binding_array<sampler,>;
        @group(0) @binding(2) var outputs: binding_array<texture_storage_2d<rgba8unorm, write>, 3>;
        @compute @workgroup_size(1) fn run(@builtin(workgroup_id) group: vec3u) {
            let value = textureSampleLevel(images[group.x], states[0], vec2f(0.5), 0.0);
            textureStore(outputs[2], vec2u(0), value);
        }
        """
    let module = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: textureArrayCapabilities).module
    let expected: [(IRTypeKind, IRArraySize)] = [
        (.sampledFloat2D, .constant(3)),
        (.sampler(comparison: false), .dynamic),
        (.image(dimension: .d2, arrayed: false, class: .storage(format: .rgba8Unorm, access: .store)), .constant(3)),
    ]
    #expect(module.globalVariables.count == expected.count)
    for (global, (element, size)) in zip(module.globalVariables.values, expected) {
        guard case .bindingArray(let base, let count) = try module.types.value(for: global.type).kind else {
            Issue.record("Expected a binding array type")
            continue
        }
        #expect(global.space == .handle)
        #expect(count == size)
        #expect(try module.types.value(for: base).kind == element)
    }
}

@Test func wgslBindingArraysIndexBufferElementsInTheirAddressSpace() throws {
    let source = """
        enable wgpu_binding_array;
        struct Uniform { index: u32 }
        struct Record { head: u32, tail: array<u32> }
        override count: u32 = 4;
        @group(0) @binding(0) var<uniform> selectors: binding_array<Uniform, count>;
        @group(0) @binding(1) var<storage, read_write> records: binding_array<Record>;
        @compute @workgroup_size(1) fn run() {
            let index = selectors[0].index;
            let length = arrayLength(&records[index].tail);
            records[index].head = length;
        }
        """
    let module = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: bufferArrayCapabilities).module
    let uniform = module.globalVariables.values[0]
    let storage = module.globalVariables.values[1]
    guard case .bindingArray(_, .override(let count)) = try module.types.value(for: uniform.type).kind,
          case .bindingArray(let record, .dynamic) = try module.types.value(for: storage.type).kind else {
        Issue.record("Expected a pipeline-sized uniform array and a runtime-sized storage array")
        return
    }
    #expect(try module.overrides.value(for: count).name == "count")
    #expect(module.functions.values[0].expressions.values.contains { expression in
        guard case .access = expression.kind else { return false }
        return (try? module.types.value(for: expression.type).kind) == .pointer(base: record, space: storage.space)
    })
}

@Test(arguments: [
    ("@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 1>;", "", ShaderDiagnosticCode.unsupportedCapability),
    ("enable wgpu_binding_array; @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 0>;", "", .invalidType),
    ("enable wgpu_binding_array; @group(0) @binding(0) var<storage> images: binding_array<u32, 1>;", "", .invalidType),
    ("enable wgpu_binding_array; @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 1>;",
     "_ = textureDimensions(images[1]);", .typeMismatch),
    ("enable wgpu_binding_array; @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;",
     "_ = textureDimensions(images[-1]);", .typeMismatch),
    ("enable wgpu_binding_array; @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 1>;",
     "_ = textureDimensions(images[true]);", .typeMismatch),
    ("enable wgpu_binding_array; @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 1>;",
     "_ = images;", .typeMismatch),
    ("enable wgpu_binding_array; @group(0) @binding(0) var<uniform> images: binding_array<texture_2d<f32>, 1>;",
     "", .invalidAddressSpace),
    ("enable wgpu_binding_array; struct Buffer { data: u32 } var<workgroup> images: binding_array<Buffer, 1>;",
     "", .invalidAddressSpace),
])
func wgslBindingArraysRejectInvalidDeclarationsAndIndices(declaration: String, body: String, code: ShaderDiagnosticCode) {
    let source = ShaderInput.wgsl("\(declaration) @compute @workgroup_size(1) fn run() { \(body) }")
    #expect(throws: ShaderFailure.self) {
        do {
            _ = try IndependentShaderCompiler().parseAndValidate(source, capabilities: textureArrayCapabilities.union(bufferArrayCapabilities))
        } catch let error as ShaderFailure {
            if case .diagnostics(let diagnostics) = error {
                #expect(diagnostics.first?.code == code)
            } else {
                Issue.record("Expected a typed diagnostic, received \(error)")
            }
            throw error
        }
    }
}

@Test(arguments: [
    ("var images: binding_array<texture_2d<f32>, 2>;", "_ = textureDimensions(images[index]);",
     ShaderCapabilities.textureAndSamplerBindingArray, ShaderCapabilities.textureAndSamplerBindingArrayNonUniformIndexing),
    ("var images: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;",
     "textureStore(images[index], vec2u(0), vec4f(1));",
     .storageTextureBindingArray, .storageTextureBindingArrayNonUniformIndexing),
    ("var<uniform> buffers: binding_array<Record, 2>;", "_ = buffers[index].word;",
     .bufferBindingArray, .bufferBindingArrayNonUniformIndexing),
    ("var<storage, read_write> buffers: binding_array<Record, 2>;", "buffers[index].word = 1;",
     .storageBufferBindingArray, .storageBufferBindingArrayNonUniformIndexing),
])
func wgslBindingArrayCapabilitiesAreCheckedDuringSemanticValidation(
    declaration: String, body: String, resourceCapability: ShaderCapabilities, indexingCapability: ShaderCapabilities
) throws {
    let source = """
        enable wgpu_binding_array;
        struct Record { word: u32 }
        @group(0) @binding(0) \(declaration)
        @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) index: u32) {
            \(body)
        }
        """
    let capabilities = resourceCapability.union(indexingCapability)
    let module = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: capabilities).module
    #expect(throws: ShaderFailure.self) {
        try ShaderSemanticValidator(capabilities: resourceCapability).validate(module)
    }
    #expect(throws: ShaderFailure.self) {
        try ShaderSemanticValidator(capabilities: indexingCapability).validate(module)
    }
}

@Test func wgslBindingArrayAcceptsUniformIndicesFromReadOnlyStorage() throws {
    func source(access: String) -> ShaderInput {
        .wgsl("""
            enable wgpu_binding_array;
            @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
            @group(0) @binding(1) var<storage, \(access)> selector: u32;
            @compute @workgroup_size(2) fn run() {
                _ = textureDimensions(images[selector]);
            }
            """)
    }
    let compiler = IndependentShaderCompiler()
    let capabilities = ShaderCapabilities.textureAndSamplerBindingArray
    _ = try compiler.parseAndValidate(source(access: "read"), capabilities: capabilities)
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(source(access: "read_write"), capabilities: capabilities)
    }
}

private let compiledBindingArrayShader = """
    enable wgpu_binding_array;
    override count: u32 = 2;
    @group(0) @binding(0) var first: texture_2d<f32>;
    @group(0) @binding(1) var images: binding_array<texture_2d<f32>, count>;
    @group(0) @binding(2) var last: texture_2d<f32>;
    @group(1) @binding(0) var states: binding_array<sampler, 2>;
    @group(1) @binding(1) var targets: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
    @group(2) @binding(0) var<storage, read_write> result: array<u32, 4>;

    fn sample(t: texture_2d<f32>, s: sampler) -> f32 {
        return textureSampleLevel(t, s, vec2f(1.25, 0.5), 0).x;
    }

    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32) {
        let index = i32(lane) - 1;
        let sampled = sample(images[index], states[index]);
        result[lane] = u32(round(sampled * 255.0)) + textureDimensions(first).x + textureDimensions(last).x;
        if lane == 1u || lane == 2u {
            textureStore(targets[index], vec2u(0), vec4f(sampled, 0, 0, 1));
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .wgsl])
func computeBindingArrayWritersEmitSpecializedResourceArrays(format: ShaderOutputFormat) throws {
    let input = ShaderInput.wgsl(compiledBindingArrayShader)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(input, to: format, constants: ["count": 3])
    let images = try #require(shader.sampledTextures.first { $0.binding.binding == 1 })
    #expect(images.bindingArray?.count == 3)
    #expect(images.samplingSamplers == [.init(group: 1, binding: 0)])
    #expect(shader.samplerBindings.first?.bindingArray?.count == 2)
    #expect(shader.storageTextures.first?.bindingArray?.count == 2)
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    if format == .msl {
        #expect(source.contains("array<texture2d<float, access::sample>, 3>"))
        #expect(source.contains("array<sampler, 2>"))
        #expect(source.contains("array<texture2d<float, access::write>, 2>"))
        #expect(source.contains("min("))
    } else if format == .hlsl {
        let registerBase = try #require(images.bindingArray?.hlslRegisterBase)
        #expect(registerBase > 2)
        #expect(source.contains("[3] : register(t\(registerBase), space0)"))
        #expect(source.contains("SamplerState saga_sampler_3[2]"))
        #expect(source.contains("saga_texture_4[2]"))
        #expect(source.contains("NonUniformResourceIndex("))
    } else {
        #expect(source.hasPrefix("enable wgpu_binding_array;\n"))
        #expect(source.contains("binding_array<texture_2d<f32>, 3>"))
        #expect(source.contains("binding_array<sampler, 2>"))
        #expect(source.contains("binding_array<texture_storage_2d<rgba8unorm, write>, 2>"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: textureArrayCapabilities.union([
            .textureAndSamplerBindingArrayNonUniformIndexing, .storageTextureBindingArrayNonUniformIndexing,
        ]))
    }
}

#if canImport(Metal)
@Test func computeBindingArraysDispatchDistinctMetalResources() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(compiledBindingArrayShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let sampleUsage: TextureUsage = [.textureBinding, .copyDestination]
    let storeUsage: TextureUsage = [.storageBinding, .copySource]
    let a = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: sampleUsage))
    let b = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: sampleUsage))
    let first = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: storeUsage))
    let second = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: storeUsage))
    let clamp = try await device.createSampler(.init())
    let repeating = try await device.createSampler(.init(addressModeU: .repeat))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let region = TextureCopyRegion(width: 2, height: 1)
    let textures: [ComputeBindingSlot: ResourceID] = [
        .init(binding: 0): a, .init(binding: 2): b,
        .init(binding: 1, arrayElement: 0): a, .init(binding: 1, arrayElement: 1): b,
        .init(group: 1, binding: 1, arrayElement: 0): first, .init(group: 1, binding: 1, arrayElement: 1): second,
    ]
    let samplers: [ComputeBindingSlot: ResourceID] = [
        .init(group: 1, binding: 0, arrayElement: 0): clamp, .init(group: 1, binding: 0, arrayElement: 1): repeating,
    ]
    let bindings = [ComputeBindingSlot(group: 2, binding: 0): output]
    var missing = textures
    missing.removeValue(forKey: .init(binding: 1, arrayElement: 1))
    await #expect(throws: GPUError.self) {
        try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1), textures: missing, samplers: samplers),
        ])
    }
    let words = try await device.submitCommands([
        .writeTexture(destination: a, data: [10, 0, 0, 255, 70, 0, 0, 255], region: region),
        .writeTexture(destination: b, data: [150, 0, 0, 255, 220, 0, 0, 255], region: region),
        .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1), textures: textures, samplers: samplers),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ]).wait()
    #expect(words.flatMap { $0 } == [74, 74, 154, 154, 0xff000046, 0xff000096])
    await device.close()
}
#endif

@Test func computeBindingArrayWritersRejectInvalidSpecializedCounts() throws {
    let compiler = IndependentShaderCompiler()
    let shader = ShaderInput.wgsl(compiledBindingArrayShader)
    for value: Double in [0, 25] {
        #expect(throws: ShaderFailure.self) {
            try compiler.compileCompute(shader, to: .msl, constants: ["count": value])
        }
    }
}
