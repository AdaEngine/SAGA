import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private func float32TextureShader(filter: Bool, coordinates: String = "vec2f(0.5)") -> ShaderInput {
    let sampler = filter ? "@group(0) @binding(3) var state: sampler;" : ""
    func read(_ name: String) -> String {
        filter ? "textureSampleLevel(\(name), state, \(coordinates), 0.0)" : "textureLoad(\(name), vec2u(0), 0u)"
    }
    return .wgsl("""
        @group(0) @binding(0) var single: texture_2d<f32>;
        @group(0) @binding(1) var pair: texture_2d<f32>;
        @group(0) @binding(2) var color: texture_2d<f32>;
        \(sampler)
        @group(0) @binding(4) var<storage, read_write> out: vec4f;
        fn readAll() -> vec4f {
            return vec4f(\(read("single")).x, \(read("pair")).y, \(read("color")).w, f32(textureDimensions(single).x));
        }
        @compute @workgroup_size(1) fn run() { out = readAll(); }
        """)
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func reflectsFilteringOnlyForSampledFloatBindings(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for filter in [false, true] {
        let shader = try compiler.compileCompute(float32TextureShader(filter: filter), to: format)
        #expect(shader.sampledTextures.count == 3)
        #expect(shader.sampledTextures.allSatisfy { $0.requiresFiltering == filter })
    }
    let mixed = try compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var loaded: texture_2d<f32>;
        @group(0) @binding(1) var sampled: texture_2d<f32>;
        @group(0) @binding(2) var state: sampler;
        @group(0) @binding(3) var<storage, read_write> out: vec4f;
        @compute @workgroup_size(1) fn run() {
            out = textureLoad(loaded, vec2u(0), 0u) + textureSampleLevel(sampled, state, vec2f(0.5), 0.0);
        }
        """), to: format)
    let usage = Dictionary(uniqueKeysWithValues: mixed.sampledTextures.map { ($0.binding.binding, $0.requiresFiltering) })
    #expect(usage == [0: false, 1: true])
}

@Test(arguments: [TextureFormat.r32Float, .rg32Float, .rgba32Float])
func float32TextureLoadsAreCoreAndSamplingRequiresFilterability(format: TextureFormat) throws {
    let texture = TextureDescriptor(width: 2, height: 1, format: format, usage: [.textureBinding, .copyDestination])
    try texture.validate(against: .init(maxBufferSize: 64))
    let image = ResourceID(device: 1, serial: 1)
    let slot = ComputeBindingSlot(binding: 0)
    for requiresFiltering in [false, true] {
        let pipeline = ComputePipeline(id: .init(device: 1, serial: 2), storageBindings: [],
            textureBindings: [.init(slot: slot, sampleType: .float, requiresFiltering: requiresFiltering)])
        let command = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1), textures: [slot: image])
        func check(features: Set<GPUFeature>) throws {
            try command.validate(device: 1, descriptors: [:], textures: [image: texture], samplerDescriptors: [:], textureViews: [:],
                                 features: features)
        }
        try check(features: [.float32Filterable])
        if requiresFiltering {
            #expect(throws: GPUError.validation("Float32 sampling requires float32-filterable")) { try check(features: []) }
        } else {
            try check(features: [])
        }
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func explicitNonFilteringSamplersApplyToEveryTexturePair(format: ShaderOutputFormat) throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        @group(1) @binding(3) var single: texture_2d<f32>;
        @group(1) @binding(4) var shared: texture_2d<f32>;
        @group(0) @binding(2) var nearest: sampler;
        @group(1) @binding(1) var other: sampler;
        @group(0) @binding(0) var<storage, read_write> out: vec4f;
        fn read() -> vec4f {
            return textureSampleLevel(single, nearest, vec2f(0.5), 0.0)
                + textureSampleLevel(shared, nearest, vec2f(0.5), 0.0)
                + textureSampleLevel(shared, other, vec2f(0.5), 0.0);
        }
        @compute @workgroup_size(1) fn run() { out = read(); }
        """), to: format)
    let id = ResourceID(device: 1, serial: 1)
    let primary = ComputeBindingSlot(binding: 2)
    let alternate = ComputeBindingSlot(group: 1, binding: 1)
    for selection: Set<ComputeBindingSlot> in [[], [primary], [primary, alternate]] {
        let pipeline = try ComputePipeline(id: id, shader: shader, nonFilteringSamplers: selection)
        let textures = Dictionary(uniqueKeysWithValues: pipeline.textureBindings.map { ($0.slot.binding, $0.requiresFiltering) })
        #expect(textures[3] == !selection.contains(primary))
        #expect(textures[4] == !(selection.contains(primary) && selection.contains(alternate)))
        #expect(pipeline.nonFilteringSamplers == selection)
    }
    #expect(throws: GPUError.validation("Non-filtering selection must name a non-comparison sampler binding")) {
        try ComputePipeline(id: id, shader: shader, nonFilteringSamplers: [.init(group: 1, binding: 3)])
    }
}

@Test func explicitNonFilteringSelectionRejectsComparisonSamplers() throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_depth_2d;
        @group(0) @binding(1) var state: sampler_comparison;
        @group(0) @binding(2) var<storage, read_write> out: f32;
        @compute @workgroup_size(1) fn run() {
            out = textureSampleCompareLevel(image, state, vec2f(0.5), 0.5);
        }
        """), to: .wgsl)
    #expect(throws: GPUError.validation("Non-filtering selection must name a non-comparison sampler binding")) {
        try ComputePipeline(id: .init(device: 1, serial: 1), shader: shader, nonFilteringSamplers: [.init(binding: 1)])
    }
}

@Test func renderStagesMergeFloat32LoadAndFilteringRequirements() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var state: sampler;
        @vertex fn vs() -> @builtin(position) vec4f { return textureLoad(image, vec2u(0), 0u); }
        @fragment fn fs() -> @location(0) vec4f { return textureSampleLevel(image, state, vec2f(0.5), 0.0); }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    #expect(vertex.sampledTextures.first?.requiresFiltering == false)
    #expect(fragment.sampledTextures.first?.requiresFiltering == true)
    let pipeline = try RenderPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm,
                                      vertexBuffers: [], vertex: vertex, fragment: fragment)
    #expect(pipeline.sampledTextures.first?.requirement.requiresFiltering == true)
    let nearest = try RenderPipeline(id: .init(device: 1, serial: 2), colorFormat: .rgba8Unorm,
                                     vertexBuffers: [], vertex: vertex, fragment: fragment,
                                     nonFilteringSamplers: [.init(binding: 1)])
    #expect(nearest.sampledTextures.first?.requirement.requiresFiltering == false)
    #expect(nearest.samplerBindings.first?.requirement.requiresNonFiltering == true)
}

#if canImport(Metal)
@Test func loadsAndFiltersFloat32TexturesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let formats: [TextureFormat] = [.r32Float, .rg32Float, .rgba32Float]
    let values: [[Float]] = [[1, 3], [1, 10, 3, 14], [1, 2, 3, 0.25, 4, 5, 6, 0.75]]
    var textures: [ComputeBindingSlot: ResourceID] = [:]
    var uploads: [BufferCommand] = []
    for (index, format) in formats.enumerated() {
        let image = try await device.createTexture(.init(width: 2, height: 1, format: format,
                                                         usage: [.copyDestination, .textureBinding]))
        textures[.init(binding: UInt32(index))] = image
        let bytes = values[index].withUnsafeBytes { Array($0) }
        uploads.append(.writeTexture(destination: image, data: bytes, region: .init(width: 2, height: 1)))
    }
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let sampler = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let compiler = IndependentShaderCompiler()
    var results: [[UInt32]] = []
    for filter in [false, true] {
        let shader = try compiler.compileCompute(float32TextureShader(filter: filter), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let dispatch = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 4): output],
            workgroups: .init(x: 1), textures: textures, samplers: filter ? [.init(binding: 3): sampler] : [:])
        if filter && !device.capabilities.features.contains(.float32Filterable) {
            await #expect(throws: GPUError.validation("Float32 sampling requires float32-filterable")) {
                try await device.submitCommands([dispatch])
            }
            break
        }
        let capture: [BufferCommand] = [
            dispatch, .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]
        results += try await device.submitCommands(filter ? capture : capture + uploads + capture).wait()
    }
    let nearestShader = try compiler.compileCompute(
        float32TextureShader(filter: true, coordinates: "vec2f(0.75, 0.5)"), to: .msl)
    let samplerSlot = ComputeBindingSlot(binding: 3)
    let nearest = try await device.createComputePipeline(nearestShader, nonFilteringSamplers: [samplerSlot])
    let nearestSampler = try await device.createSampler(.init())
    func dispatch(sampler: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: nearest, bindings: [.init(binding: 4): output], workgroups: .init(x: 1),
                         textures: textures, samplers: [samplerSlot: sampler])
    }
    let invalid = dispatch(sampler: sampler)
    await #expect(throws: GPUError.validation("Invalid sampler binding")) {
        try await device.submitCommands([invalid])
    }
    results += try await device.submitCommands([
        dispatch(sampler: nearestSampler), .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]).wait()
    let expected: [[Float]] = device.capabilities.features.contains(.float32Filterable) ?
        [[0, 0, 0, 2], [1, 10, 0.25, 2], [2, 12, 0.5, 2]] : [[0, 0, 0, 2], [1, 10, 0.25, 2]]
    #expect(results == (expected + [[3, 14, 0.75, 2]]).map { $0.map(\.bitPattern) })
    let image = try #require(textures[.init(binding: 0)])
    let pixels = try await renderNearestFloat32(device, image: image, sampler: nearestSampler)
    #expect(pixels == [0xff00bf00])
    await device.close()
}

private func renderNearestFloat32(_ device: Device, image: ResourceID, sampler: ResourceID) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var state: sampler;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            let z = 0.25 * textureLoad(image, vec2u(0), 0u).x;
            if i == 0u { return vec4f(-1, -1, z, 1); }
            if i == 1u { return vec4f(3, -1, z, 1); }
            return vec4f(-1, 3, z, 1);
        }
        @fragment fn fs() -> @location(0) vec4f {
            let green = 0.25 * textureSampleLevel(image, state, vec2f(0.75, 0.5), 0.0).x;
            return vec4f(0, green, 0, 1);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let samplerSlot = ComputeBindingSlot(binding: 1)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                         nonFilteringSamplers: [samplerSlot])
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    return try await device.submitCommands([
        .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): image], samplers: [samplerSlot: sampler]),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]).wait().flatMap { $0 }
}
#endif
