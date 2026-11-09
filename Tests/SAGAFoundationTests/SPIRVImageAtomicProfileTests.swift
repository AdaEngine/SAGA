import SAGACompiler
import SAGAShaderIR
import Testing

@Test func spirvImageAtomicProfileRequiresNativeOutputOnlyForTheAtomicEntry() throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var atomicImage: texture_storage_2d<r32uint, atomic>;
        @group(0) @binding(1) var portableImage: texture_storage_2d<r32uint, read_write>;

        @compute @workgroup_size(1) fn nativeAtomic() {
            textureAtomicAdd(atomicImage, vec2i(0), 7u);
        }
        @compute @workgroup_size(1) fn portable() {
            let coordinate = vec2i(0);
            let previous = textureLoad(portableImage, coordinate);
            textureStore(portableImage, coordinate, previous + vec4u(11u, 0u, 0u, 0u));
        }
        """)
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let input = ShaderInput.spirv(words)

    let native = try compiler.compileCompute(input, entryPoint: "nativeAtomic", to: .wgsl)
    try #require(native.storageTextures.count == 1)
    let atomicImage = try #require(native.storageTextures.first)
    #expect(atomicImage.binding == .init(group: 0, binding: 0))
    #expect(atomicImage.format == .r32Uint)
    #expect(atomicImage.access == [.load, .store, .atomic])
    #expect(throws: ShaderFailure.unsupported("Atomic and R64Uint storage textures require native shader output")) {
        try compiler.compileWebGPUCompute(input, entryPoint: "nativeAtomic")
    }

    let portable = try compiler.compileWebGPUCompute(input, entryPoint: "portable")
    try #require(portable.storageTextures.count == 1)
    let portableImage = try #require(portable.storageTextures.first)
    #expect(portableImage.binding == .init(group: 0, binding: 1))
    #expect(portableImage.format == .r32Uint)
    #expect(portableImage.access == [.load, .store])
    guard case .text(.wgsl, let text) = portable.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("textureLoad("))
    #expect(text.contains("textureStore("))
}
