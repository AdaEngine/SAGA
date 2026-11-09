import SAGACompiler
import SAGAShaderIR
import Testing

@Test func spirvR64ImageProfileRejectsOnlyTheWideEntry() throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var wideSource: texture_storage_2d<r64uint, read>;
        @group(0) @binding(1) var wideDestination: texture_storage_2d<r64uint, write>;
        @group(0) @binding(2) var portableImage: texture_storage_2d<r32uint, read_write>;

        @compute @workgroup_size(1) fn wideCopy(@builtin(global_invocation_id) id: vec3u) {
            let coordinate = vec2u(id.x, 0u);
            textureStore(wideDestination, coordinate, textureLoad(wideSource, coordinate));
        }
        @compute @workgroup_size(1) fn portable() {
            let coordinate = vec2u(0u);
            let previous = textureLoad(portableImage, coordinate);
            textureStore(portableImage, coordinate, previous + vec4u(11u, 0u, 0u, 0u));
        }
        """)
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let input = ShaderInput.spirv(words)

    let native = try compiler.compileCompute(input, entryPoint: "wideCopy", to: .msl)
    #expect(native.storageTextures.map(\.binding) == [.init(group: 0, binding: 0), .init(group: 0, binding: 1)])
    #expect(native.storageTextures.map(\.format) == [.r64Uint, .r64Uint])
    #expect(native.storageTextures.map(\.access) == [.load, .store])
    #expect(native.requiresShaderInt64)
    do {
        _ = try compiler.compileWebGPUCompute(input, entryPoint: "wideCopy")
        Issue.record("WebGPU accepted the R64Uint entry")
    } catch ShaderFailure.unsupported(let message) {
        #expect(message == "64-bit integer types require native shader output" ||
                message == "Atomic and R64Uint storage textures require native shader output")
    }

    for portableInput in [source, input] {
        let portable = try compiler.compileWebGPUCompute(portableInput, entryPoint: "portable")
        #expect(portable.storageTextures.map(\.binding) == [.init(group: 0, binding: 2)])
        #expect(portable.storageTextures.map(\.format) == [.r32Uint])
        #expect(portable.storageTextures.map(\.access) == [[.load, .store]])
        #expect(!portable.requiresShaderInt64)
        #expect(!portable.requiresShaderInt64AtomicMinMax)
        #expect(!portable.requiresShaderInt64AtomicAllOps)
        guard case .text(.wgsl, let text) = portable.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(text.contains("textureLoad("))
        #expect(text.contains("textureStore("))
        #expect(!text.contains("r64uint"))
    }
}
