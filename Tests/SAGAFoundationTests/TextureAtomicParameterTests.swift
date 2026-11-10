import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let textureAtomicParameterSource = """
        @group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>;
        fn increment(target: texture_storage_2d<r32uint, atomic>) {
            textureAtomicAdd(target, vec2i(0), 1u);
        }
        @compute @workgroup_size(4) fn main() { increment(image); }
        """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func textureAtomicThroughFunctionParameter(_ format: ShaderOutputFormat) throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(textureAtomicParameterSource), to: format)
    #expect(shader.storageTextures.count == 1)
    guard case .text(_, let output) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    switch format {
    case .msl: #expect(output.contains(".atomic_fetch_add("))
    case .hlsl: #expect(output.contains("InterlockedAdd("))
    case .glsl: #expect(output.contains("imageAtomicAdd("))
    default: Issue.record("Unexpected target")
    }
}

#if canImport(Metal)
@Test(.enabled("Metal device must support texture atomics") {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let supported = device.capabilities.features.contains(.textureAtomic)
    await device.close()
    return supported
})
func textureAtomicParameterExecutesThroughMetalDevice() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(textureAtomicParameterSource), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let texture = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint,
            usage: [.storageAtomic, .copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.mapRead, .copyDestination]))
        let words = try await device.submitCommands([
            .writeTexture(destination: texture, data: [10, 0, 0, 0, 37, 0, 0, 0],
                          region: .init(width: 2, height: 1)),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): texture]),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(words == [[10, 37], [14, 37]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
