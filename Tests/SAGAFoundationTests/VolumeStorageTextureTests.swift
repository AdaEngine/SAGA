import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let volumeStorageShader = """
    @group(0) @binding(0) var source: texture_storage_3d<r32uint, read>;
    @group(0) @binding(1) var image: texture_storage_3d<r32uint, read_write>;
    @group(0) @binding(2) var<storage, read_write> result: array<vec4u, 5>;
    @compute @workgroup_size(1) fn run() {
        let input = textureLoad(source, vec3u(0, 0, 2));
        let previous = textureLoad(image, vec3i(0, 0, 1));
        result[0] = vec4u(textureDimensions(image), input.x);
        result[1] = previous;
        result[2] = textureLoad(source, vec3u(0, 0, 4));
        result[3] = textureLoad(image, vec3i(0, 0, -1));
        result[4] = textureLoad(image, vec3u(0, 0, 0xffffffffu));
        textureStore(image, vec3i(0, 0, 1), previous + input);
        textureStore(image, vec3u(1, 0, 0), input + vec4u(1u));
        textureStore(image, vec3i(0, 0, -1), vec4u(99u));
        textureStore(image, vec3u(0, 0, 2), vec4u(99u));
        textureStore(image, vec3u(0, 0, 0xffffffffu), vec4u(99u));
    }
    """

private let volumeStorageFragment = """
    @group(1) @binding(3) var volume: texture_storage_3d<rgba8unorm, write>;
    @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        if index == 0u { return vec4f(-1, -1, 0, 1); }
        if index == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> @location(0) vec4f {
        textureStore(volume, vec3u(0, 0, 1), vec4f(1, 0, 1, 1));
        return vec4f(0, 1, 0, 1);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsVolumeStorageLoadsAndStores(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let compute = try compiler.compileCompute(.wgsl(volumeStorageShader), to: format)
    let fragment = try compiler.compileRenderStage(.wgsl(volumeStorageFragment), entryPoint: "fs", to: format)
    #expect(compute.storageTextures.map(\.access) == [.load, [.load, .store]])
    #expect(fragment.storageTextures.map(\.access) == [.store])
    #expect(compute.storageTextures.map(\.dimension) == [.d3, .d3])
    #expect(fragment.storageTextures.map(\.dimension) == [.d3])
    guard case .text(_, let source) = compute.artifact, case .text(_, let draw) = fragment.artifact else {
        Issue.record("Expected source shaders"); return
    }
    switch format {
    case .wgsl:
        #expect(source.contains("texture_storage_3d<r32uint, read_write>"))
        _ = try compiler.parseAndValidate(.wgsl(source))
        _ = try compiler.parseAndValidate(.wgsl(draw))
    case .msl:
        #expect(source.contains("texture3d<uint, access::read_write>"))
        #expect(source.contains(".get_depth()") && draw.contains("texture3d<float, access::write>"))
    case .hlsl:
        #expect(source.contains("RWTexture3D<uint>") && source.contains(".GetDimensions("))
        #expect(draw.contains("RWTexture3D<unorm float4>"))
    case .glsl:
        #expect(source.contains("readonly uniform uimage3D") && source.contains("imageLoad("))
        #expect(draw.contains("writeonly uniform image3D") && draw.contains("imageStore("))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    "let value = textureLoad(image, vec2u(0));",
    "let value = textureLoad(image, vec3u(0), 0u);",
    "let size = textureDimensions(image, 0u);",
    "let layers = textureNumLayers(image);",
    "textureStore(image, vec2u(0), vec4u(0));",
    "textureStore(source, vec3u(0), vec4u(0));",
])
func rejectsInvalidVolumeStorageOperands(body: String) throws {
    let source = """
        @group(0) @binding(0) var source: texture_storage_3d<r32uint, read>;
        @group(0) @binding(1) var image: texture_storage_3d<r32uint, read_write>;
        @compute @workgroup_size(1) fn run() { \(body) }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid volume storage access accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func writesVolumeStorageMipsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compiler = IndependentShaderCompiler()
        let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(volumeStorageShader), to: .msl))
        let shader = ShaderInput.wgsl(volumeStorageFragment)
        let draw = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(shader, entryPoint: "vs", to: .msl),
            fragment: compiler.compileRenderStage(shader, entryPoint: "fs", to: .msl), colorFormat: .rgba8Unorm)
        let volume = try await device.createTexture(.init(width: 4, height: 2, format: .r32Uint,
            usage: [.storageBinding, .copySource, .copyDestination], mipLevelCount: 2, dimension: .d3, depth: 4))
        let source = try await device.createTextureView(texture: volume, .init(mipLevelCount: 1))
        let image = try await device.createTextureView(texture: volume, .init(baseMipLevel: 1))
        let plane = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint, usage: .storageBinding))
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
            usage: [.storageBinding, .copySource], dimension: .d3, depth: 2))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
        let upload = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 80, usage: [.mapRead, .copyDestination]))
        func dispatch(_ input: ResourceID, _ outputImage: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): input, .init(binding: 1): outputImage])
        }
        await #expect(throws: GPUError.validation("Storage texture view dimension does not match the shader binding")) {
            try await device.submitCommands([dispatch(source, plane)])
        }
        await #expect(throws: GPUError.validation("Storage texture binding requires one mip level of the declared format")) {
            try await device.submitCommands([dispatch(volume, image)])
        }
        await #expect(throws: GPUError.self) {
            try await device.submitCommands([dispatch(image, image)])
        }
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [7, 20, 21]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: volume,
                                 region: .init(width: 1, height: 1, z: 2)),
            .copyBufferToTexture(source: upload, layout: .init(offset: 4), destination: volume,
                                 region: .init(width: 2, height: 1, mipLevel: 1, z: 1)),
            dispatch(source, image),
            .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
            .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, mipLevel: 1), .read(buffer: read, byteCount: 8),
            .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, mipLevel: 1, z: 1), .read(buffer: read, byteCount: 8),
            .render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(group: 1, binding: 3): color]),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256, z: 1), .read(buffer: read, byteCount: 4),
        ])
        let expected: [[UInt32]] = [[2, 1, 2, 7, 20, 0, 0, 1] + Array(repeating: 0, count: 12),
                                    [0, 8], [27, 21], [0], [0xffff00ff]]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
