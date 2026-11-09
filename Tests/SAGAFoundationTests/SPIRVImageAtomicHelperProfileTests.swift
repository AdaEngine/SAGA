import SAGACompiler
import SAGAShaderIR
import Testing

@Test func wgslImageAtomicHelpersRequireAGlobalTextureBinding() throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var atomicImage: texture_storage_2d<r32uint, atomic>;

        fn update(image: texture_storage_2d<r32uint, atomic>) {
            textureAtomicAdd(image, vec2i(0), 7u);
        }
        @compute @workgroup_size(1) fn run() {
            update(atomicImage);
        }
        """)
    let compiler = IndependentShaderCompiler()
    do {
        _ = try compiler.parseAndValidate(source, capabilities: .referenceDefault.union(.textureAtomic))
        Issue.record("WGSL accepted an image atomic on a helper parameter")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains {
            $0.code == .invalidReference && $0.message == "Texture atomics require a global texture binding"
        })
    }
}

@Test func spirvImageAtomicHelperProfilesFollowTheForwardedBinding() throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.imageAtomicHelperModule(includePortable: true))
    let compiler = IndependentShaderCompiler()
    let native = try compiler.compileCompute(input, entryPoint: "atomic", to: .msl)
    #expect(native.storageTextures.map(\.binding) == [.init(group: 0, binding: 0)])
    #expect(native.storageTextures.first?.access == [.load, .store, .atomic])
    #expect(throws: ShaderFailure.unsupported("Atomic and R64Uint storage textures require native shader output")) {
        try compiler.compileWebGPUCompute(input, entryPoint: "atomic")
    }

    let portable = try compiler.compileWebGPUCompute(input, entryPoint: "portable")
    #expect(portable.storageTextures.map(\.binding) == [.init(group: 0, binding: 1)])
    #expect(portable.storageTextures.first?.access == [.load, .store])
    guard case .text(.wgsl, let text) = portable.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("textureLoad("))
    #expect(text.contains("textureStore("))
}
