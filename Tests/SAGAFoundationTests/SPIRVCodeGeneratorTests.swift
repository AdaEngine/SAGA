import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvStorageSource = ShaderInput.wgsl("""
    @group(1) @binding(2) var<storage, read> source: array<i32, 4>;
    @group(0) @binding(3) var<storage, read_write> result: array<u32, 4>;
    fn encode(a: i32, b: i32) -> u32 { return bitcast<u32>(a + b * -3i); }
    @compute @workgroup_size(1) fn processStorage() {
        var a: i32;
        a += source[1u];
        let b = source[2u];
        let index = bitcast<u32>(source[0u]);
        result[0u] = encode(a, b);
        result[1u] = bitcast<u32>(a - b);
        result[2u] = bitcast<u32>(b) + 0x80000000u;
        result[index] = bitcast<u32>(a) * 3u + 7u;
    }
    """)

private func spirvWords(_ artifact: ShaderArtifact) throws -> [UInt32] {
    guard case .spirv(let words) = artifact else { throw ShaderFailure.incompatibleArtifact }
    return words
}

@Test func spirvCodeGeneratorEmitsValidatedComputeBinaryAndPreservesReflection() throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compile(spirvStorageSource, to: .spirv)
    let words = try spirvWords(artifact)
    try #require(words.count > 5)
    #expect(words[0] == 0x07230203)
    #expect(words[3] > 0)
    #expect(try compiler.compile(spirvStorageSource, to: .spirv) == artifact)
    let input = ShaderInput.spirv(words)
    let module = try compiler.parseAndValidate(input).module
    let entry = try #require(module.entryPoints.first)
    #expect(module.entryPoints.count == 1)
    #expect(entry.name == "processStorage")
    #expect(entry.stage == .compute)
    #expect(entry.workgroupSize == .init(x: 1))
    let shader = try compiler.compileCompute(input, entryPoint: "processStorage", to: .wgsl)
    #expect(shader.workgroupSize == .init(x: 1))
    let source = try #require(shader.storageBindings.first { $0.binding == .init(group: 1, binding: 2) })
    let destination = try #require(shader.storageBindings.first { $0.binding == .init(group: 0, binding: 3) })
    #expect(shader.storageBindings.count == 2)
    #expect(source.isReadOnly && !destination.isReadOnly)
    #expect(source.minimumByteSize == 16 && destination.minimumByteSize == 16)
    guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.wgsl(text))
}

@Test func spirvCodeGeneratorRejectsMalformedShaderAndBinaryWithDiagnostics() throws {
    let compiler = IndependentShaderCompiler()
    let invalid = ShaderInput.wgsl("@compute @workgroup_size(1) fn run() { let value: u32 = true; }")
    do {
        _ = try compiler.compile(invalid, to: .spirv)
        Issue.record("SPIR-V export accepted malformed WGSL")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .typeMismatch && $0.span != nil })
    }
    var words = try spirvWords(compiler.compile(spirvStorageSource, to: .spirv))
    try #require(!words.isEmpty)
    words[0] = 0
    do {
        _ = try compiler.parseAndValidate(.spirv(words))
        Issue.record("SPIR-V decoder accepted a corrupt magic word")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .invalidBinary })
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvCodeGeneratorRoundTripsDepthOnlyFragment(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl("""
        @fragment fn writeDepth() -> @builtin(frag_depth) f32 { return 0.5; }
        """)
    let words = try spirvWords(compiler.compile(input, to: .spirv))
    let fragment = try compiler.compileRenderStage(.spirv(words), entryPoint: "writeDepth", to: format)
    #expect(fragment.writesDepth && fragment.colorOutputComponentCounts.isEmpty)
}

#if canImport(Metal)
@Test func spirvCodeGeneratorRoundTripsIntegerStorageOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let words = try spirvWords(compiler.compile(spirvStorageSource, to: .spirv))
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "processStorage", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let destination = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: source, offset: 0, words: [3, 0xfffffff7, 4, 0xcafebabe]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(group: 1, binding: 2): source, .init(binding: 3): destination],
                             workgroups: .init(x: 1)),
            .copy(source: destination, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [[0xffffffeb, 0xfffffff3, 0x80000004, 0xffffffec]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
