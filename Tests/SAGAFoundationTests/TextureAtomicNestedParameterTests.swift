import Testing
import SAGACompiler
import SAGAShaderIR
import SAGACore
import SAGAMetal
import SAGATypes
#if canImport(Metal)
import Metal
#endif

private let nestedAtomicTexture = """
    enable wgpu_binding_array;

    @group(0) @binding(0) var images: binding_array<texture_storage_2d<r32uint, atomic>, 2>;
    fn inner(image: texture_storage_2d<r32uint, atomic>) {
        textureAtomicAdd(image, vec2i(0), 1u);
    }
    fn outer(image: texture_storage_2d<r32uint, atomic>) { inner(image); }
    @compute @workgroup_size(1) fn main() { outer(images[1]); }
    """

@Test func nestedTextureAtomicParameterRequiresBothCapabilities() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl(nestedAtomicTexture), capabilities: [.textureAtomic])
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl(nestedAtomicTexture), capabilities: [.storageTextureBindingArray])
    }
    _ = try compiler.parseAndValidate(.wgsl(nestedAtomicTexture), capabilities: [.textureAtomic, .storageTextureBindingArray])
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func nestedTextureAtomicParameterFromBindingArray(format: ShaderOutputFormat) throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(nestedAtomicTexture), to: format)
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    #expect(source.contains("saga_fn_"))
    switch format {
    case .msl: #expect(source.contains(".atomic_fetch_add("))
    case .hlsl: #expect(source.contains("InterlockedAdd("))
    case .glsl: #expect(source.contains("imageAtomicAdd("))
    default: Issue.record("Unexpected target")
    }
}

#if canImport(Metal)
@Test(.enabled("Metal device must support texture atomics and Metal 3") {
    guard MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true else { return false }
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let supported = device.capabilities.features.contains(.textureAtomic)
    await device.close()
    return supported
})
func nestedTextureAtomicBindingArrayMetalReadback() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(nestedAtomicTexture), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let first = try await device.createTexture(.init(width: 1, height: 1, format: .r32Uint,
            usage: [.storageAtomic, .copyDestination, .copySource]))
        let second = try await device.createTexture(.init(width: 1, height: 1, format: .r32Uint,
            usage: [.storageAtomic, .copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
        let words = try await device.submitCommands([
            .writeTexture(destination: first, data: [10, 0, 0, 0], region: .init(width: 1, height: 1)),
            .writeTexture(destination: second, data: [37, 0, 0, 0], region: .init(width: 1, height: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                textures: [.init(binding: 0, arrayElement: 0): first, .init(binding: 0, arrayElement: 1): second]),
            .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(words == [[10], [38]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
