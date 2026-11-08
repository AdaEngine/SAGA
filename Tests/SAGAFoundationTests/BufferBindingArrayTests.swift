import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let bufferBindingArrayShader = """
    enable wgpu_binding_array;
    requires unrestricted_pointer_parameters;
    override count: u32 = 2;
    struct Settings { values: vec4u }
    struct Header { marker: u32 }
    struct Record { header: Header, tail: array<u32> }
    @group(0) @binding(1) var<storage, read_write> records: binding_array<Record, count>;
    @group(0) @binding(2) var<uniform> settings: binding_array<Settings, count>;
    @group(0) @binding(3) var<storage, read_write> output: array<vec4u, 4>;

    fn loadWord(word: ptr<storage, u32, read_write>) -> u32 { return *word; }

    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let index = i32(lane) - 1;
        let length = arrayLength(&records[index].tail);
        let value = records[index].tail[2];
        let first = loadWord(&records[index].tail[0]);
        let marker = records[index].header.marker + settings[index].values.x;
        output[lane] = vec4u(length, value, marker, first);
        records[index].tail[0] = settings[index].values.y;
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .wgsl])
func computeBufferBindingArraysReflectElementLayouts(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(bufferBindingArrayShader), to: format, constants: ["count": 3])
    let records = try #require(shader.storageBindings.first { $0.binding.binding == 1 })
    let settings = try #require(shader.uniformBindings.first)
    #expect(records.bindingArray?.count == 3)
    #expect(records.minimumByteSize == 8)
    #expect(records.runtimeArrayStride == 4)
    #expect(records.runtimeArrayOffset == 4)
    #expect(settings.bindingArray?.count == 3)
    #expect(settings.minimumByteSize == 16)
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    switch format {
    case .msl:
        #expect(shader.metalRuntimeLengthBuffer != nil)
        #expect(source.contains("array<device uint*, 3>"))
        #expect(source.contains("array<constant uint*, 3>"))
    case .hlsl:
        let storageRegister = try #require(records.bindingArray?.hlslRegisterBase)
        let uniformRegister = try #require(settings.bindingArray?.hlslRegisterBase)
        #expect(source.contains("saga_buffer_0[3] : register(u\(storageRegister), space0)"))
        #expect(source.contains("saga_buffer_1[3] : register(b\(uniformRegister), space0)"))
        #expect(source.contains(".GetDimensions(bytes)"))
        #expect(source.contains("NonUniformResourceIndex("))
    case .wgsl:
        #expect(source.contains("enable wgpu_binding_array;"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union([
            .bufferBindingArray, .bufferBindingArrayNonUniformIndexing,
            .storageBufferBindingArray, .storageBufferBindingArrayNonUniformIndexing,
        ]))
    default: Issue.record("Unexpected output format")
    }
}

#if canImport(Metal)
@Test func computeBufferBindingArraysSelectMetalRangesAndLengths() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(bufferBindingArrayShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let records = try await device.createBuffer(.init(size: 768, usage: [.storage, .copyDestination, .copySource]))
    let settings = try await device.createBuffer(.init(size: 512, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
    let bindings: [ComputeBindingSlot: StorageBufferBinding] = [
        .init(binding: 1, arrayElement: 0): .init(buffer: records, offset: 256, size: 8),
        .init(binding: 1, arrayElement: 1): .init(buffer: records, offset: 512, size: 16),
        .init(binding: 2, arrayElement: 0): .init(buffer: settings, size: 16),
        .init(binding: 2, arrayElement: 1): .init(buffer: settings, offset: 256, size: 16),
        .init(binding: 3): .init(buffer: output),
    ]
    var aliased = bindings
    aliased[.init(binding: 1, arrayElement: 1)] = aliased[.init(binding: 1, arrayElement: 0)]
    await #expect(throws: GPUError.validation("Aliased writable compute buffers")) {
        try await device.submitCommands([.dispatchComputeRanges(pipeline: pipeline, bindings: aliased, workgroups: .init(x: 1))])
    }
    let words = try await device.submitCommands([
        .write(buffer: records, offset: 256, words: [100, 5, 101, 102]),
        .write(buffer: records, offset: 512, words: [200, 7, 8, 9]),
        .write(buffer: settings, offset: 0, words: [11, 55, 0, 0]),
        .write(buffer: settings, offset: 256, words: [22, 77, 0, 0]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        .copy(source: records, sourceOffset: 256, destination: readback, destinationOffset: 0, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
        .copy(source: records, sourceOffset: 512, destination: readback, destinationOffset: 0, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]).wait()
    #expect(words == [[0, 0, 0, 0, 1, 0, 111, 5, 3, 9, 222, 7, 0, 0, 0, 0],
                      [100, 55, 101, 102], [200, 77, 8, 9]])
    await device.close()
}
#endif

private let renderBufferBindingArrayShader = """
    enable wgpu_binding_array;
    struct Settings { values: vec4u }
    struct Record { tail: array<u32> }
    struct Varyings {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(flat) red: f32,
    }
    @group(0) @binding(0) var<uniform> extra: Settings;
    @group(0) @binding(2) var<uniform> settings: binding_array<Settings, 2>;
    @group(1) @binding(0) var<storage, read_write> records: binding_array<Record, 2>;

    @vertex fn vs(@builtin(vertex_index) index: u32) -> Varyings {
        let xy = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        let red = f32(extra.values.x + settings[0].values.x);
        return Varyings(vec4f(xy[index], 0, 1), red);
    }

    @fragment fn fs(input: Varyings) -> @location(0) vec4f {
        let index = u32(input.position.x);
        let length = f32(arrayLength(&records[index].tail));
        let previous = f32(records[index].tail[0]);
        records[index].tail[0] = settings[index].values.y;
        return vec4f(vec3f(input.red, length, previous) / 255.0, 1);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .wgsl])
func renderBufferBindingArraysReflectBothStages(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(renderBufferBindingArrayShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
    let vertexSettings = try #require(vertex.uniformBindings.first { $0.binding.binding == 2 })
    let fragmentSettings = try #require(fragment.uniformBindings.first)
    #expect(vertexSettings.bindingArray?.count == 2)
    #expect(vertexSettings.bindingArray == fragmentSettings.bindingArray)
    #expect(fragment.storageBindings.first?.bindingArray?.count == 2)
    if format == .msl {
        #expect(vertexSettings.flatBindingIndex != fragmentSettings.flatBindingIndex)
        #expect(fragment.metalRuntimeLengthBuffer != nil)
    }
}

#if canImport(Metal)
@Test func renderBufferBindingArraysSelectMetalRanges() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(renderBufferBindingArrayShader)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .msl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .msl), colorFormat: .rgba8Unorm)
    let settings = try await device.createBuffer(.init(size: 512, usage: [.uniform, .copyDestination]))
    let records = try await device.createBuffer(.init(size: 512, usage: [.storage, .copyDestination, .copySource]))
    let target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let words = try await device.submitCommands([
        .write(buffer: settings, offset: 0, words: [11, 55, 0, 0]),
        .write(buffer: settings, offset: 256, words: [22, 77, 0, 0]),
        .write(buffer: records, offset: 0, words: [5, 42]),
        .write(buffer: records, offset: 256, words: [7, 8, 9, 42]),
        .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                uniforms: [.init(binding: 0): .init(buffer: settings, size: 16),
                           .init(binding: 2, arrayElement: 0): .init(buffer: settings, size: 16),
                           .init(binding: 2, arrayElement: 1): .init(buffer: settings, offset: 256, size: 16)],
                storageBuffers: [.init(group: 1, binding: 0, arrayElement: 0): .init(buffer: records, size: 4),
                                 .init(group: 1, binding: 0, arrayElement: 1): .init(buffer: records, offset: 256, size: 12)]),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        .copy(source: records, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
        .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]).wait()
    #expect(words == [[0xff050116, 0xff070316], [55, 42], [77, 8, 9, 42]])
    await device.close()
}
#endif
