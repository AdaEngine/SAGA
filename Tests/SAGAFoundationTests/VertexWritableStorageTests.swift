import SAGABackend
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
#endif

private let vertexStorageShader = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read_write> visited: atomic<u32>;
    @group(0) @binding(1) var values: texture_storage_2d<r32uint, write>;
    @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        _ = atomicOr(&visited, 1u << index);
        textureStore(values, vec2u(index, 0u), vec4u(index + 5u, 0u, 0u, 1u));
        return vec4f(points[index], 0, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    """)

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func nativeVertexWritableStoragePreservesBindingsThroughSPIRV(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(vertexStorageShader, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    for input in [vertexStorageShader, .spirv(words)] {
        let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: format)
        #expect(vertex.storageBindings.map(\.isReadOnly) == [false])
        #expect(vertex.storageTextures.map(\.access) == [.store])
        #expect(throws: ShaderFailure.unsupported("Browser WebGPU does not support vertex writable storage")) {
            try compiler.compileWebGPURenderStage(input, entryPoint: "vs")
        }
        let fragment = try compiler.compileWebGPURenderStage(input, entryPoint: "fs")
        #expect(fragment.storageBindings.isEmpty && fragment.storageTextures.isEmpty)
    }
}

@Test func nativeVertexWritableStorageRequiresAdvertisedDeviceFeature() throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(vertexStorageShader, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(vertexStorageShader, entryPoint: "fs", to: .msl)
    let unsupported = try DeviceCapabilities(maxBufferSize: 4096)
    let supported = try DeviceCapabilities(maxBufferSize: 4096, features: [.vertexWritableStorage])
    #expect(throws: GPUError.self) {
        try RenderPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm, vertexBuffers: [],
                           vertex: vertex, fragment: fragment, capabilities: unsupported)
    }
    _ = try RenderPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm, vertexBuffers: [],
                           vertex: vertex, fragment: fragment, capabilities: supported)
}

#if canImport(Metal)
@Test func nativeVertexWritableStorageWritesBufferAndTextureOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(vertexStorageShader, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(vertexStorageShader, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let visited = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let values = try await device.createTexture(.init(width: 3, height: 1, format: .r32Uint,
            usage: [.storageBinding, .copySource]))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 1): values], storageBuffers: [.init(binding: 0): .init(buffer: visited)]),
            .copy(source: visited, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: values, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 12),
            .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[7], [5, 6, 7], [0xff00ff00]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
