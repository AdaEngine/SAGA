import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let textureInt64AtomicSource = """
    @group(0) @binding(0) var image: texture_storage_2d<r64uint, atomic>;
    fn maximum(value: u64) {
        textureAtomicMax(image, vec2u(0, 0), value);
    }
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        maximum(0xfedcba9876543210lu);
        if lane == 0u { textureAtomicMin(image, vec2u(1, 0), 10376293541461622785lu); }
        textureAtomicMax(image, vec2i(-1, 0), 18446744073709551615lu);
    }
    """

private let textureInt64CopySource = """
    @group(0) @binding(0) var input: texture_storage_2d<r64uint, read>;
    @group(0) @binding(1) var output: texture_storage_2d<r64uint, write>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
        let pixel = vec2u(lane, 0u);
        textureStore(output, pixel, textureLoad(input, pixel));
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func textureInt64AtomicWritersPreserveUnsignedWidth(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(textureInt64AtomicSource), to: format)
    #expect(shader.storageTextures.first?.format == .r64Uint)
    #expect(shader.storageTextures.first?.access == [.load, .store, .atomic])
    guard case .text(_, let text) = shader.artifact else {
        Issue.record("Expected shader text")
        return
    }
    #expect(text.contains("18364758544493064720"))
    #expect(text.contains("18446744073709551615"))
    let copy = try compiler.compileCompute(.wgsl(textureInt64CopySource), to: format)
    #expect(copy.storageTextures.map(\.format) == [.r64Uint, .r64Uint])
    switch format {
    case .msl:
        #expect(text.contains("texture2d<ulong, access::read_write>"))
        #expect(text.contains(".atomic_min("))
    case .hlsl:
        #expect(shader.hlslProfile == "cs_6_6")
        #expect(copy.hlslProfile == "cs_6_6")
        #expect(text.contains("RWTexture2D<uint64_t>"))
        #expect(text.contains("InterlockedMin("))
    case .glsl:
        #expect(text.contains("GL_EXT_shader_image_int64"))
        #expect(text.contains("u64image2D"))
        #expect(text.contains("imageAtomicMin("))
    case .wgsl:
        #expect(text.contains("r64uint, atomic"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union([.shaderInt64, .textureInt64Atomic]))
        if case .text(.wgsl, let copyText) = copy.artifact {
            _ = try compiler.parseAndValidate(.wgsl(copyText), capabilities: .referenceDefault.union([.shaderInt64, .textureInt64Atomic]))
        }
    default: Issue.record("Unexpected output format")
    }
}

#if canImport(Metal)
@Test func textureInt64AtomicRetainsHighBitsThroughMetalDevice() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    guard device.capabilities.features.contains(.textureInt64Atomic) else { return }
    let compiler = IndependentShaderCompiler()
    let atomic = try await device.createComputePipeline(compiler.compileCompute(.wgsl(textureInt64AtomicSource), to: .msl))
    let copy = try await device.createComputePipeline(compiler.compileCompute(.wgsl(textureInt64CopySource), to: .msl))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .r64Uint,
        usage: [.storageAtomic, .storageBinding, .copyDestination]))
    let output = try await device.createTexture(.init(width: 2, height: 1, format: .r64Uint,
        usage: [.storageBinding, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let bytes = [UInt64(0x8000_0000_0000_0001), .max].flatMap { value in
        withUnsafeBytes(of: value.littleEndian) { Array($0) }
    }
    let words = try await device.submitCommands([
        .writeTexture(destination: image, data: bytes, region: .init(width: 2, height: 1)),
        .dispatchCompute(pipeline: atomic, bindings: [:], workgroups: .init(x: 1), textures: [.init(binding: 0): image]),
        .dispatchCompute(pipeline: copy, bindings: [:], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): image, .init(binding: 1): output]),
        .copyTextureToBuffer(source: output, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 16),
    ]).wait()
    #expect(words == [[0x7654_3210, 0xfedc_ba98, 1, 0x9000_0000]])
    await device.close()
}
#endif

@Test func textureInt64AtomicRejectsWrongWidthAndOperation() throws {
    let compiler = IndependentShaderCompiler()
    let capabilities = ShaderCapabilities.referenceDefault.union([.shaderInt64, .textureInt64Atomic])
    for operation in [
        "textureAtomicAdd(image, vec2u(0), 1lu)",
        "textureAtomicMax(image, vec2u(0), 1u)",
        "textureAtomicMin(image, vec2u(0), 18446744073709551616lu)",
    ] {
        let source = "@group(0) @binding(0) var image: texture_storage_2d<r64uint, atomic>; @compute @workgroup_size(1) fn run() { \(operation); }"
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(source), capabilities: capabilities) }
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl(textureInt64AtomicSource), capabilities: .referenceDefault.union([.shaderInt64, .textureAtomic]))
    }
}

@Test func textureInt64AtomicWebTargetRejectsNativeUnsignedValues() throws {
    let source = ShaderInput.wgsl("""
        fn observe(value: u64) {}
        @compute @workgroup_size(1) fn run() { observe(18446744073709551615lu); }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileCompute(source, to: .wgsl)
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(source) }
}

@Test func textureInt64AtomicHLSLRenderRequiresShaderModel66() throws {
    let shader = try IndependentShaderCompiler().compileRenderStage(.wgsl("""
        @group(0) @binding(0) var image: texture_storage_2d_array<r64uint, atomic>;
        @fragment fn fs() -> @location(0) vec4f {
            textureAtomicMin(image, vec2u(0), 1u, 18446744073709551615lu);
            return vec4f(0.0);
        }
        """), entryPoint: "fs", to: .hlsl)
    #expect(shader.hlslProfile == "ps_6_6")
    #expect(shader.storageTextures.first?.isArrayed == true)
}
