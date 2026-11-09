import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let imageShader = ShaderInput.glsl("""
    #version 450
    layout(local_size_x=1) in;
    layout(r32ui, binding=0) readonly uniform uimage2D sourceImage;
    layout(r32ui, binding=1) writeonly uniform uimage2D targetImage;
    void main() {
        ivec2 size = imageSize(targetImage);
        uvec4 value = imageLoad(sourceImage, ivec2(0));
        imageStore(targetImage, ivec2(0), value + uvec4(uint(size.x)));
    }
    """, stage: .compute, defines: [:])

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func glslStorageImagePreservesBindingsAndAccess(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(imageShader, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    for input in [imageShader, .spirv(words)] {
        let shader = try compiler.compileCompute(input, to: format)
        #expect(shader.storageTextures.map(\.format) == [.r32Uint, .r32Uint])
        #expect(shader.storageTextures.map(\.access) == [.load, .store])
        if format == .wgsl, case .text(.wgsl, let source) = shader.artifact {
            _ = try compiler.parseAndValidate(.wgsl(source))
        }
    }
}

@Test func glslStorageImageAllowsSizeForQueryOnlyImage() throws {
    let shader = ShaderInput.glsl("""
        #version 450
        layout(rgba8, binding=0) readonly writeonly uniform image2DArray queryImage;
        void main() { ivec3 dimensions = imageSize(queryImage); }
        """, stage: .compute, defines: [:])
    _ = try IndependentShaderCompiler().compileWebGPUCompute(shader)
}

@Test(arguments: [
    ("readonly", "imageStore(image, ivec2(0), uvec4(0));"),
    ("writeonly", "uvec4 value = imageLoad(image, ivec2(0));"),
    ("readonly writeonly", "uvec4 value = imageLoad(image, ivec2(0));"),
    ("", "imageStore(image, ivec3(0), uvec4(0));"),
])
func glslStorageImageRejectsInvalidAccessAndCoordinates(_ qualifier: String, _ body: String) {
    let source = "#version 450\nlayout(r32ui,binding=0) \(qualifier) uniform uimage2D image; void main() { \(body) }"
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.glsl(source, stage: .compute, defines: [:]))
    }
}

#if canImport(Metal)
@Test func glslStorageImageReadsAndWritesExactTexelOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(imageShader, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createTexture(.init(width: 1, height: 1, format: .r32Uint,
                                                          usage: [.storageBinding, .copyDestination]))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .r32Uint,
                                                          usage: [.storageBinding, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .writeTexture(destination: source, data: [41, 0, 0, 0], region: .init(width: 1, height: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): source, .init(binding: 1): target]),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[42]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
