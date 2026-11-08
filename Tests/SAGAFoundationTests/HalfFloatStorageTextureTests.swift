import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGABackend
import SAGAMetal

private let halfStorageWriteShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var single: texture_storage_2d<r16float, write>;
    @group(0) @binding(1) var pair: texture_storage_2d<rg16float, write>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) x: u32) {
        let p = vec2u(x, 0u);
        textureStore(single, p, vec4f(0.5 + f32(x), 99, 99, 99));
        textureStore(pair, p, vec4f(-2 + f32(x), 0.25 + f32(x), 99, 99));
    }
    """

private let halfStorageReadShader = """
    requires texture_formats_tier1, readonly_and_readwrite_storage_textures;
    @group(0) @binding(0) var single: texture_storage_2d<r16float, read>;
    @group(0) @binding(1) var pair: texture_storage_2d<rg16float, read>;
    @group(0) @binding(2) var<storage, read_write> out: array<vec4f, 2>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) x: u32) {
        let p = vec2u(x, 0u);
        let rg = textureLoad(pair, p);
        out[x] = vec4f(textureLoad(single, p).x, rg.xy, rg.w);
    }
    """

private let halfStorageReadWriteShader = """
    requires texture_formats_tier1, readonly_and_readwrite_storage_textures;
    @group(0) @binding(0) var single: texture_storage_2d<r16float, read_write>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) x: u32) {
        let p = vec2u(x, 0u);
        textureStore(single, p, textureLoad(single, p) + vec4f(2, 0, 0, 0));
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func halfFloatStorageFormatsReflectAccessAndCapabilities(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let write = try compiler.compileCompute(.wgsl(halfStorageWriteShader), to: format)
    let read = try compiler.compileCompute(.wgsl(halfStorageReadShader), to: format)
    let update = try compiler.compileCompute(.wgsl(halfStorageReadWriteShader), to: format)
    #expect(write.storageTextures.map(\.format) == [.r16Float, .rg16Float])
    #expect(write.storageTextures.map(\.access) == [.store, .store])
    #expect(read.storageTextures.map(\.access) == [.load, .load])
    #expect(update.storageTextures.map(\.access) == [[.load, .store]])
    #expect(write.storageTextureFeatures == [.halfFloatStorageTextures])
    #expect(read.storageTextureFeatures == [.halfFloatStorageTextures, .storageTextureRead])
    #expect(update.storageTextureFeatures == [.halfFloatStorageTextures, .r16FloatStorageReadWrite, .storageTextureReadWrite])
    for shader in [write, read, update] {
        guard case .text(_, let code) = shader.artifact else { Issue.record("Expected shader source"); return }
        if format == .wgsl {
            #expect(code.contains("requires texture_formats_tier1;"))
            _ = try compiler.parseAndValidate(.wgsl(code))
        }
        if format == .glsl {
            #expect(code.contains("layout(r16f,"))
            #expect(code.contains("layout(rg16f,") == (shader.storageTextures.count == 2))
        }
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func fragmentHalfStorageBindingsRetainTheirFormatAndFeature(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        requires texture_formats_tier1;
        @group(0) @binding(0) var image: texture_storage_2d<rg16float, write>;
        @fragment fn fs() -> @location(0) vec4f {
            textureStore(image, vec2i(0), vec4f(0.5));
            return vec4f(1);
        }
        """)
    let shader = try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
    #expect(shader.storageTextures.map(\.format) == [.rg16Float])
    #expect(shader.storageTextures.requiredStorageTextureFeatures == [.halfFloatStorageTextures])
    if format == .wgsl, case .text(_, let code) = shader.artifact {
        #expect(code.contains("requires texture_formats_tier1;"))
        _ = try compiler.parseAndValidate(.wgsl(code))
    }
}

@Test func narrowHalfStorageTextureDescriptorsRequireTheDeviceFeature() throws {
    let baseline = try DeviceCapabilities(maxBufferSize: 128)
    let available = try DeviceCapabilities(maxBufferSize: 128, features: [.halfFloatStorageTextures])
    for format: TextureFormat in [.r16Float, .rg16Float] {
        let descriptor = TextureDescriptor(width: 4, height: 2, format: format,
                                           usage: [.storageBinding, .copySource, .textureBinding])
        #expect(throws: GPUError.validation("Feature not advertised")) { try descriptor.validate(against: baseline) }
        try descriptor.validate(against: available)
    }
}

#if canImport(Metal)
@Test func readsWritesAndUpdatesNarrowHalfStorageTexturesOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.halfFloatStorageTextures])
    let words = try await halfStorageDiagnostic(device)
    let expected: [Float] = [0.5, -2, 0.25, 1, 1.5, -1, 1.25, 1]
    let updated: [Float] = [2.5, -2, 0.25, 1, 3.5, -1, 1.25, 1]
    #expect(words == (expected + (device.capabilities.features.contains(.r16FloatStorageReadWrite) ? updated : [])).map(\.bitPattern))
    await device.close()
}

private func halfStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let write = try await device.createComputePipeline(compiler.compileCompute(.wgsl(halfStorageWriteShader), to: .msl))
    let read = try await device.createComputePipeline(compiler.compileCompute(.wgsl(halfStorageReadShader), to: .msl))
    let single = try await device.createTexture(.init(width: 2, height: 1, format: .r16Float, usage: .storageBinding))
    let pair = try await device.createTexture(.init(width: 2, height: 1, format: .rg16Float, usage: .storageBinding))
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
    let images: [ComputeBindingSlot: ResourceID] = [.init(binding: 0): single, .init(binding: 1): pair]
    let groups = DispatchWorkgroups(x: 1)
    let capture: [BufferCommand] = [
        .dispatchCompute(pipeline: read, bindings: [.init(binding: 2): output], workgroups: groups, textures: images),
        .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
    ]
    var commands: [BufferCommand] = [
        .dispatchCompute(pipeline: write, bindings: [:], workgroups: groups, textures: images),
    ] + capture
    if device.capabilities.features.contains(.r16FloatStorageReadWrite) {
        let update = try await device.createComputePipeline(compiler.compileCompute(.wgsl(halfStorageReadWriteShader), to: .msl))
        commands += [.dispatchCompute(pipeline: update, bindings: [:], workgroups: groups, textures: [.init(binding: 0): single])]
        commands += capture
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}
#endif
