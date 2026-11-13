import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let storageTextureShader = """
    @group(2) @binding(3) var source: texture_2d<f32>;
    @group(2) @binding(4) var destination: texture_storage_2d<rgba8unorm, write>;
    fn writeColor(x: i32, color: vec4f) { textureStore(destination, vec2i(x, 0), color); }
    @compute @workgroup_size(5) fn run(@builtin(global_invocation_id) id: vec3u) {
        let x = i32(id.x) - 1;
        if id.x == 4u {
            textureStore(destination, vec2u(0xffffffffu, 0u), vec4f(1, 1, 0, 1));
        } else if x != 0 {
            let dimensions = textureDimensions(destination);
            let red = textureLoad(source, vec2u(0), 0).r;
            writeColor(x, vec4f(red, 0.0, f32(dimensions.x - 1u), 1.0));
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsGuardedWriteOnlyStorageTextures(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(storageTextureShader), to: format)
    #expect(shader.sampledTextures.map(\.binding) == [.init(group: 2, binding: 3)])
    let binding = try #require(shader.storageTextures.first)
    #expect(shader.storageTextures.count == 1)
    #expect(binding.binding == .init(group: 2, binding: 4))
    #expect(binding.format == .rgba8Unorm)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("texture_storage_2d<rgba8unorm, write>"))
        #expect(source.contains("textureStore(") && source.contains("textureDimensions("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains("access::write> saga_texture_1 [[texture(24)]]"))
        #expect(source.contains(".write(") && source.contains(".get_width()"))
    case .hlsl:
        #expect(source.contains("RWTexture2D<unorm float4>") && source.contains("register(u4, space2)"))
        #expect(source.contains(".GetDimensions("))
    case .glsl:
        #expect(source.contains("layout(rgba8, binding=0) writeonly uniform image2D"))
        #expect(source.contains("imageStore(") && source.contains("imageSize("))
    default: Issue.record("Unexpected format")
    }
}

@Test(arguments: [
    "textureStore(image, vec2f(0), vec4f(1));",
    "textureStore(image, vec3i(0), vec4f(1));",
    "textureStore(image, vec2i(0), vec4u(1));",
    "textureStore(sampled, vec2i(0), vec4f(1));",
    "let size = textureDimensions(image, 0);",
])
func rejectsInvalidStorageTextureOperations(body: String) throws {
    let source = """
        @group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
        @group(0) @binding(1) var sampled: texture_2d<f32>;
        @compute @workgroup_size(1) fn run() { \(body) }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid storage texture operation accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .typeMismatch)
    }
}

#if canImport(Metal)
@Test func writesStorageMipWithoutClobberingOtherTexelsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(storageTextureShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
            usage: [.copySource, .copyDestination, .textureBinding, .storageBinding], mipLevelCount: 2))
        let source = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
        let destination = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
        let sampledOnly = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: .textureBinding))
        let upload = try await device.createBuffer(.init(size: 272, usage: [.copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
        let inputSlot = ComputeBindingSlot(group: 2, binding: 3)
        let outputSlot = ComputeBindingSlot(group: 2, binding: 4)
        func dispatch(_ input: ResourceID, _ output: ResourceID) -> BufferCommand {
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                             textures: [inputSlot: input, outputSlot: output])
        }
        await #expect(throws: GPUError.validation("Aliased writable compute textures")) {
            try await device.submitCommands([dispatch(destination, destination)])
        }
        for output in [image, sampledOnly] {
            await #expect(throws: GPUError.self) {
                try await device.submitCommands([dispatch(source, output)])
            }
        }
        let red = Array(repeating: UInt32(0xff0000ff), count: 4)
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: red),
            .write(buffer: upload, offset: 256, words: red),
            .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: image,
                                 region: .init(width: 4, height: 2)),
            dispatch(source, destination),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 8),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 272),
        ])
        #expect(try await ticket.wait() == [[0, 0xffff00ff], red + Array(repeating: 0, count: 60) + red])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
