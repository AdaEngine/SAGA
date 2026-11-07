import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let vectorStorageShader = """
    @group(0) @binding(1) var<storage, read> src: array<vec3u>;
    @group(0) @binding(3) var<storage, read_write> dst: array<vec3i>;
    @group(0) @binding(5) var<storage, read_write> pair: array<vec2f, 1>;
    fn read(i: i32) -> vec3u { return src[i]; }
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        dst[id.x] = vec3i(read(i32(id.x) - 1i)) +
            vec3i(i32(arrayLength(&src)), i32(arrayLength(&dst)), -1i);
        if id.x == 0u {
            dst[id.x].z = -7i;
            let value = pair[0u];
            pair[0u] = vec2f(value.y, value.x);
        }
        if id.x == 1u { dst[id.x][i32(id.x) - 2i] = -9i; }
        if id.x == 2u { dst[id.x][id.x - 1u] = dst[id.x].x + 100i; }
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsVectorStorageUsingWGSLStride(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let compiled = try compiler.compileCompute(.wgsl(vectorStorageShader), to: format)
    #expect(compiled.storageBindings.map(\.minimumByteSize) == [16, 16, 8])
    #expect(compiled.storageBindings.map(\.runtimeArrayStride) == [16, 16, nil])
    guard case .text(_, let source) = compiled.artifact else { Issue.record("Expected text shader"); return }
    switch format {
    case .msl:
        #expect(source.contains("saga_lengths [[buffer(8)]]"))
    case .hlsl:
        #expect(source.contains("GetDimensions("))
    case .glsl, .wgsl:
        let input: ShaderInput = format == .glsl ? .glsl(source, stage: .compute, defines: [:]) : .wgsl(source)
        let module = try compiler.parseAndValidate(input).module
        let strides = try module.globalVariables.values.filter { $0.binding != nil }.map { global in
            var kind = try module.types.value(for: global.type).kind
            if case .structure(let fields, _) = kind, let field = fields.first {
                kind = try module.types.value(for: field.type).kind
            }
            guard case .array(_, _, let stride) = kind else {
                throw ShaderFailure.unsupported("Expected a storage array")
            }
            return stride
        }
        #expect(strides == [16, 16, 8])
    default: Issue.record("Unexpected target")
    }
}

#if canImport(Metal)
@Test func vectorStoragePreservesPaddingAndClipsPartialElementsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(vectorStorageShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 304, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 336, usage: [.storage, .copyDestination, .copySource]))
        let pair = try await device.createBuffer(.init(size: 8, usage: [.storage, .copyDestination, .copySource]))
        let read = try await device.createBuffer(.init(size: 336, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 256, words: [10, 20, 30, 801, 40, 50, 60, 802, 777, 778, 779, 803]),
            .write(buffer: output, offset: 256, words: [
                0, 0, 0, 701, 0, 0, 0, 702, 0, 0, 0, 703, 0, 0, 0, 704, 901, 902, 903, 904,
            ]),
            .write(buffer: pair, offset: 0, words: [Float(-1.25).bitPattern, Float(3.5).bitPattern]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 1): .init(buffer: input, offset: 256, size: 36),
                .init(binding: 3): .init(buffer: output, offset: 256, size: 68),
                .init(binding: 5): .init(buffer: pair),
            ], workgroups: .init(x: 5)),
            .copy(source: output, destination: read, byteCount: 336),
            .read(buffer: read, byteCount: 336),
            .copy(source: pair, destination: read, byteCount: 8),
            .read(buffer: read, byteCount: 8),
        ])
        let words = try await ticket.wait()
        let expected: [UInt32] = [
            2, 4, UInt32(bitPattern: -7), 701, 12, 24, 29, 702, 42, 142, 59, 703,
            2, 4, .max, 704, 901, 902, 903, 904,
        ]
        #expect(words == [Array(repeating: 0, count: 64) + expected, [Float(3.5).bitPattern, Float(-1.25).bitPattern]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
