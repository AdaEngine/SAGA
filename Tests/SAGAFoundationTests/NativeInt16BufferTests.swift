import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let nativeInt16BufferShader = ShaderInput.wgsl("""
    enable wgpu_int16;
    struct Packed {
        signed: i16,
        unsigned: u16,
        lanes: vec2<u16>,
    }
    struct Nested {
        marker: u32,
        pairs: array<Packed, 2>,
    }
    struct Parameters {
        ignored: i16,
        amount: u16,
        bias: vec2<i16>,
    }
    @group(0) @binding(0) var<storage, read_write> packed: Packed;
    @group(0) @binding(1) var<storage, read_write> nested: Nested;
    @group(0) @binding(2) var<uniform> parameters: Parameters;
    @compute @workgroup_size(1) fn run() {
        packed.signed = i16(i32(packed.signed) + i32(parameters.bias.x));
        packed.unsigned = u16(u32(packed.unsigned) + u32(parameters.amount));
        nested.pairs[1] = packed;
        nested.pairs[0].lanes.y = u16(u32(nested.pairs[0].lanes.y) + 1000u);
    }
    """)

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func nativeInt16BuffersKeepPackedLayoutAcrossTextTargets(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileCompute(nativeInt16BufferShader, to: format)
    #expect(artifact.requiresShaderInt16)
    #expect(!artifact.requiresShaderF16)
    #expect(artifact.storageBindings.map(\.minimumByteSize) == [8, 20])
    #expect(artifact.uniformBindings.map(\.minimumByteSize) == [8])
    guard case .text(_, let source) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
    switch format {
    case .msl:
        #expect(source.contains("device short*") && source.contains("device ushort*"))
    case .hlsl:
        #expect(artifact.hlslProfile == "cs_6_2")
        #expect(source.contains("Load<int16_t>") && source.contains("Store<uint16_t>"))
    case .glsl:
        #expect(source.contains("GL_EXT_shader_explicit_arithmetic_types_int16"))
        #expect(source.contains("GL_EXT_shader_16bit_storage") && source.contains("atomicCompSwap"))
    case .wgsl:
        #expect(source.contains("enable wgpu_int16;"))
        _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderInt16])
        #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(nativeInt16BufferShader) }
    default: Issue.record("Expected text shader target")
    }
}

#if canImport(Metal)
@Test func nativeInt16BuffersPreserveNeighboringHalfwordsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(nativeInt16BufferShader, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let packed = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource, .copyDestination]))
        let nested = try await device.createBuffer(.init(size: 20, usage: [.storage, .copySource, .copyDestination]))
        let parameters = try await device.createBuffer(.init(size: 8, usage: [.uniform, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 20, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: packed, offset: 0, words: [0x0064fff0, 0x03e8000a]),
            .write(buffer: nested, offset: 0, words: [0xfeedface, 0x00c8ffff, 0x00020001, 0x00410042, 0x00050004]),
            .write(buffer: parameters, offset: 0, words: [0x0007fff9, 0x0005fff9]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): packed, .init(binding: 1): nested,
                                                             .init(binding: 2): parameters], workgroups: .init(x: 1)),
            .copy(source: packed, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
            .copy(source: nested, destination: read, byteCount: 20), .read(buffer: read, byteCount: 20),
        ]).wait()
        #expect(snapshots == [[0x006bffe9, 0x03e8000a], [0xfeedface, 0x00c8ffff, 0x03ea0001, 0x006bffe9, 0x03e8000a]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
