import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let matrixBufferShader = """
    struct Data { basis: mat2x3f, transforms: array<mat2x3f> }
    @group(0) @binding(0) var<storage, read_write> data: Data;
    @group(0) @binding(1) var<uniform> scale: mat2x2f;
    @group(0) @binding(2) var<storage, read_write> output: array<f32, 4>;
    @compute @workgroup_size(1)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        let saved = data.basis;
        data.basis = saved * scale;
        let i = id.x + 1u;
        data.transforms[i] = data.transforms[0u] * scale;
        data.transforms[i][id.x] = vec3f(21.0f, 22.0f, 23.0f);
        data.transforms[i][1u][1u] = 55.0f;
        data.transforms[id.x + 2u] = saved;
        let missing = data.transforms[id.x + 2u];
        output[0] = missing[0u].x;
        if arrayLength(&data.transforms) == 2u { output[1] = 1.0f; }
        output[2] = data.transforms[i][1u].y;
        output[3] = data.basis[1u].z;
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsMatrixBufferLayoutsAndStrides(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(matrixBufferShader), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [64, 16])
    #expect(shader.storageBindings.first?.runtimeArrayOffset == 32)
    #expect(shader.storageBindings.first?.runtimeArrayStride == 32)
    #expect(shader.uniformBindings.map(\.minimumByteSize) == [16])
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    switch format {
    case .msl: #expect(source.contains("as_type<float>") && source.contains("float2x3"))
    case .hlsl: #expect(source.contains("asfloat(") && source.contains("uint4 saga_buffer_1[1]"))
    case .glsl:
        #expect(source.contains("layout(offset = 32) mat2x3"))
        #expect(source.contains("uvec4") && source.contains("uintBitsToFloat("))
    case .wgsl:
        #expect(source.contains("array<mat2x3<f32>>"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    default: Issue.record("Unexpected target")
    }
}

#if canImport(Metal)
@Test func matrixBufferStoresPreservePaddingAndGuardColumnsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(matrixBufferShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let data = try await device.createBuffer(.init(size: 356, usage: [.storage, .copyDestination, .copySource]))
        let scale = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 356, usage: [.copyDestination, .mapRead]))
        func column(_ x: Float, _ y: Float, _ z: Float, _ padding: UInt32) -> [UInt32] {
            [x.bitPattern, y.bitPattern, z.bitPattern, padding]
        }
        let initial = [
            column(1, 2, 3, 701), column(4, 5, 6, 702),
            column(7, 8, 9, 703), column(10, 11, 12, 704),
            column(13, 14, 15, 705), column(16, 17, 18, 706), [999],
        ].flatMap { $0 }
        let expected = [
            column(2, 4, 6, 701), column(12, 15, 18, 702),
            column(7, 8, 9, 703), column(10, 11, 12, 704),
            column(21, 22, 23, 705), column(30, 55, 36, 706), [999],
        ].flatMap { $0 }
        let ticket = try await device.submitCommands([
            .write(buffer: data, offset: 256, words: initial),
            .write(buffer: scale, offset: 0, words: [Float(2).bitPattern, 0, 0, Float(3).bitPattern]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0): .init(buffer: data, offset: 256, size: 96),
                .init(binding: 1): .init(buffer: scale),
                .init(binding: 2): .init(buffer: output),
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 16),
            .read(buffer: read, byteCount: 16),
            .copy(source: data, destination: read, byteCount: 356),
            .read(buffer: read, byteCount: 356),
        ])
        let values: [Float] = [0, 1, 55, 18]
        #expect(try await ticket.wait() == [values.map(\.bitPattern), Array(repeating: UInt32(0), count: 64) + expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
