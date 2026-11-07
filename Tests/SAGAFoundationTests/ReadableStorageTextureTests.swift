import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let readableStorageShader = """
    @group(0) @binding(0) var mask: texture_storage_2d<rgba8unorm, read>;
    @group(0) @binding(1) var image: texture_storage_2d<r32uint, read_write>;
    @group(0) @binding(2) var<storage, read_write> before: array<vec4u, 4>;
    @compute @workgroup_size(4) fn run(@builtin(global_invocation_id) id: vec3u) {
        if id.x == 3u {
            before[3u] = textureLoad(image, vec2u(0xffffffffu, 0u));
        } else {
            let coords = vec2i(i32(id.x) - 1i, 0i);
            let old = textureLoad(image, coords);
            before[id.x] = old;
            if textureLoad(mask, coords).g > 0.5 {
                textureStore(image, coords, old + vec4u(5u, 99u, 99u, 99u));
            }
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsReadableStorageTextureModesAndSingleChannelResults(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(readableStorageShader), to: format)
    #expect(shader.storageTextures.map(\.format) == [.rgba8Unorm, .r32Uint])
    #expect(shader.storageTextures.map(\.access) == [.load, [.load, .store]])
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("texture_storage_2d<rgba8unorm, read>"))
        #expect(body.contains("texture_storage_2d<r32uint, read_write>"))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl:
        #expect(body.contains("texture2d<float, access::read>"))
        #expect(body.contains("texture2d<uint, access::read_write>"))
        #expect(body.contains(".read(") && body.contains(".write("))
    case .hlsl:
        #expect(body.contains("RWTexture2D<unorm float4>") && body.contains("RWTexture2D<uint>"))
        #expect(body.contains("0u, 0u, 1u)"))
    case .glsl:
        #expect(body.contains("layout(rgba8, binding=0) readonly uniform image2D"))
        #expect(body.contains("layout(r32ui, binding=1) uniform uimage2D"))
        #expect(body.contains("imageLoad("))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    "textureStore(mask, vec2u(0), vec4f(1));",
    "let texel = textureLoad(output, vec2u(0));",
    "let texel = textureLoad(mask, vec2u(0), 0u);",
    "textureStore(image, vec2u(0), vec4f(1));",
])
func rejectsInvalidReadableStorageTextureOperations(body: String) throws {
    let source = """
        @group(0) @binding(0) var mask: texture_storage_2d<rgba8unorm, read>;
        @group(0) @binding(1) var image: texture_storage_2d<r32uint, read_write>;
        @group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;
        @compute @workgroup_size(1) fn run() { \(body) }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid storage access accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func readsRGBA8MaskAndUpdatesR32UintStorageOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(readableStorageShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let mask = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: [.copyDestination, .storageBinding]))
        let image = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint,
            usage: [.copySource, .copyDestination, .storageBinding]))
        let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
        let out = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let maskSlot = ComputeBindingSlot(binding: 0)
        let imageSlot = ComputeBindingSlot(binding: 1)
        let outputSlot = ComputeBindingSlot(binding: 2)
        func dispatch(_ image: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: pipeline, bindings: [outputSlot: out], workgroups: .init(x: 1),
                             textures: [maskSlot: mask, imageSlot: image])
        }
        await #expect(throws: GPUError.self) { try await device.submitCommands([dispatch(mask)]) }
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff00ff00, 0xff0000ff, 10, 20]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: mask,
                                 region: .init(width: 2, height: 1)),
            dispatch(image),
            .copy(source: out, destination: read, byteCount: 64),
            .read(buffer: read, byteCount: 64),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
            .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: image,
                                 region: .init(width: 2, height: 1)),
            dispatch(image),
            .copy(source: out, destination: read, byteCount: 64),
            .read(buffer: read, byteCount: 64),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 8),
        ])
        let expected: [[UInt32]] = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [5, 0],
            [0, 0, 0, 0, 10, 0, 0, 1, 20, 0, 0, 1, 0, 0, 0, 0], [15, 20],
        ]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
