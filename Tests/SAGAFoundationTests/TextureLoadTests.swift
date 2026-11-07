import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let textureLoadShader = """
    @group(2) @binding(5) var image: texture_2d<f32>;
    @group(2) @binding(0) var<storage, read_write> out: array<vec4f, 7>;

    fn fetch(p: vec2i, level: i32) -> vec4f { return textureLoad(image, p, level); }

    @compute @workgroup_size(7)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        if id.x == 4u {
            out[id.x] = fetch(vec2i(0), 1i);
        } else if id.x == 5u {
            out[id.x] = textureLoad(image, vec2u(0xffffffffu, 0u), 0u);
        } else if id.x == 6u {
            out[id.x] = textureLoad(image, vec2u(0u), 0xffffffffu);
        } else {
            out[id.x] = fetch(vec2i(i32(id.x) - 1i, 0i), 0i);
        }
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsGuardedSampledTextureLoads(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(textureLoadShader), to: format)
    let binding = try #require(shader.sampledTextures.first)
    #expect(shader.sampledTextures.count == 1)
    #expect(binding.binding == .init(group: 2, binding: 5))
    #expect(binding.flatBindingIndex == (format == .msl || format == .glsl ? 0 : nil))
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected text shader"); return }
    switch format {
    case .wgsl:
        #expect(source.contains("@group(2) @binding(5)"))
        #expect(source.contains("textureDimensions("))
        #expect(source.contains("textureNumLevels("))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        #expect(source.contains("[[texture(0)]]"))
        #expect(source.contains(".get_num_mip_levels()"))
        #expect(source.contains(".read("))
    case .hlsl:
        #expect(source.contains("register(t5, space2)"))
        #expect(source.contains(".GetDimensions("))
        #expect(source.contains(".Load("))
    case .glsl:
        #expect(source.contains("layout(binding=0) uniform sampler2D"))
        #expect(source.contains("textureQueryLevels("))
        #expect(source.contains("texelFetch("))
    default:
        Issue.record("Unexpected target")
    }
}

@Test(arguments: [
    "textureLoad(image, vec2f(0), 0)",
    "textureLoad(image, vec2i(0), 0.0)",
    "textureLoad(image, vec3i(0), 0)",
    "textureLoad(image, vec2i(0), 0, 0)",
])
func rejectsInvalidSampledTextureLoadArguments(call: String) {
    let source = """
        @group(0) @binding(0) var image: texture_2d<f32>;
        @compute @workgroup_size(1) fn run() { let value = \(call); }
        """
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test(arguments: [TextureFormat.rgba8Unorm, .bgra8Unorm])
func readsUploadedAndUninitializedTexturesOnMetal(format: TextureFormat) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(textureLoadShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
        let target = try await device.createTexture(.init(width: 2, height: 1, format: format,
                                                          usage: [.copyDestination, .textureBinding]))
        let fresh = try await device.createTexture(.init(width: 2, height: 1, format: format, usage: .textureBinding))
        let unbindable = try await device.createTexture(.init(width: 2, height: 1, format: format, usage: .copySource))
        let size: UInt64 = 7 * 16
        let out = try await device.createBuffer(.init(size: size, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
        let textureSlot = ComputeBindingSlot(group: 2, binding: 5)
        let bufferSlot = ComputeBindingSlot(group: 2, binding: 0)

        func dispatch(_ texture: ResourceID, slot: ComputeBindingSlot) -> BufferCommand {
            .dispatchCompute(pipeline: pipeline, bindings: [bufferSlot: out], workgroups: .init(x: 1),
                             textures: [slot: texture])
        }
        #expect(dispatch(target, slot: textureSlot).resources.contains(target))
        for invalid in [dispatch(unbindable, slot: textureSlot), dispatch(target, slot: bufferSlot)] {
            await #expect(throws: GPUError.self) { try await device.submitCommands([invalid]) }
        }
        let red: UInt32 = format == .rgba8Unorm ? 0xff0000ff : 0xffff0000
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [red, 0xff00ff00]),
            .copyBufferToTexture(source: input, layout: .init(), destination: target, region: .init(width: 2, height: 1)),
            dispatch(target, slot: textureSlot),
            .copy(source: out, destination: read, byteCount: size),
            .read(buffer: read, byteCount: size),
            dispatch(fresh, slot: textureSlot),
            .copy(source: out, destination: read, byteCount: size),
            .read(buffer: read, byteCount: size),
        ])
        try await device.destroyTexture(target)
        let expected: [Float] = [
            0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        #expect(try await ticket.wait() == [expected.map(\.bitPattern), Array(repeating: 0, count: 28)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

@Test func sampledTextureBuiltinsPreserveUniformControlFlowInWGSL() throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @compute @workgroup_size(4) fn run() {
            let levels = textureNumLevels(image);
            let base = textureDimensions(image);
            let size = textureDimensions(image, levels - 1u);
            let pixel = textureLoad(image, vec2i(0), 0i);
            if base.x == size.x {
                if pixel.a > 0.0 { workgroupBarrier(); }
            }
        }
        """), to: .wgsl)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected text shader"); return }
    _ = try compiler.parseAndValidate(.wgsl(source))
}
