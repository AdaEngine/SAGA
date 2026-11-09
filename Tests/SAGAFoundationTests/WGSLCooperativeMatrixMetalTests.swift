import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

private let cooperativeTileSource = ShaderInput.wgsl("""
    enable wgpu_cooperative_matrix;
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(32)
    fn main() {
        let a = coopLoadT<coop_mat8x8<f32, A>>(&input[0], 10u);
        let b = coopLoad<coop_mat8x8<f32, B>>(&input[80]);
        let c = coopLoadT<coop_mat8x8<f32, C>>(&input[144]);
        let product = coopMultiplyAdd(a, b, c);
        coopStoreT(product, &output[0], 10u);

        let truncated = coopLoadT<coop_mat8x8<f32, A>>(&input[arrayLength(&input) - 1u]);
        coopStoreT(coopMultiplyAdd(truncated, b, c), &output[80]);
        coopStoreT(product, &output[arrayLength(&output) - 1u]);
    }
    """)

@Test func wgslCooperativeTilesEmitMetalIntrinsicsAndRejectWebGPU() throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(cooperativeTileSource, to: .msl)
    guard case .text(.msl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("simdgroup_load("))
    #expect(source.contains("simdgroup_multiply_accumulate("))
    #expect(source.contains("simdgroup_store("))
    #expect(shader.requiredSubgroupSize == 32)
    #expect(shader.requiresCooperativeMatrix)
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(cooperativeTileSource) }
}

@Test func wgslCooperativeMetalRejectsDecoratedBufferUntilMemorySemanticsAreLowered() {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) @coherent var<storage, read_write> buffer: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let tile = coopLoadT<coop_mat8x8<f32, C>>(&buffer[0]);
            coopStoreT(tile, &buffer[0]);
        }
        """)
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compileCompute(input, to: .msl) }
}

@Test func wgslCooperativeHelperRejectsDivergentCalls() {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> buffer: array<f32>;
        fn work() {
            let tile = coopLoadT<coop_mat8x8<f32, C>>(&buffer[0]);
            coopStoreT(tile, &buffer[0]);
        }
        @compute @workgroup_size(32)
        fn main(@builtin(local_invocation_index) lane: u32) {
            if lane == 0u { work(); }
        }
        """)
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(input, capabilities: [.cooperativeMatrix])
    }
}

@Test func wgslCooperativeHelperAcceptsUniformStoragePointers() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> buffer: array<f32>;
        fn work(pointer: ptr<storage, f32, read_write>) {
            let tile = coopLoadT<coop_mat8x8<f32, C>>(pointer);
            coopStoreT(tile, pointer);
        }
        @compute @workgroup_size(32) fn main() { work(&buffer[0]); }
        """)
    _ = try IndependentShaderCompiler().parseAndValidate(input, capabilities: [.cooperativeMatrix])
}

#if canImport(Metal)
@Test func wgslCooperativeMixedPrecisionAccumulatesOnMetal() async throws {
    let input = ShaderInput.wgsl("""
        enable f16;
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> input: array<f16>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoadT<coop_mat8x8<f16, A>>(&input[0], 0u);
            let b = coopLoadT<coop_mat8x8<f16, B>>(&input[64]);
            let c = coopLoadT<coop_mat8x8<f32, C>>(&output[0]);
            coopStoreT(coopMultiplyAdd(a, b, c), &output[0]);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    #expect(shader.requiresShaderF16)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let half = UInt32(Float16(1).bitPattern)
        let source = try await device.createBuffer(.init(size: 256, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 256, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 256, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: source, offset: 0, words: Array(repeating: half | (half << 16), count: 64)),
            .write(buffer: output, offset: 0, words: Array(repeating: Float(3).bitPattern, count: 64)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): source, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 256),
            .read(buffer: read, byteCount: 256),
        ])
        #expect(try await ticket.wait() == [Array(repeating: Float(11).bitPattern, count: 64)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func wgslCooperativeTilesMultiplyAndGuardCompleteFootprintsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(cooperativeTileSource, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        func left(_ row: Int, _ col: Int) -> Float { Float((row + 2 * col) % 5) - 2 }
        func right(_ row: Int, _ col: Int) -> Float { Float((2 * row + 3 * col) % 7) - 3 }
        func accum(_ row: Int, _ col: Int) -> Float { Float(row - col) }
        var inputValues = Array(repeating: Float(-99), count: 208)
        var expected = Array(repeating: Float(-42), count: 144)
        for row in 0..<8 {
            for column in 0..<8 {
                inputValues[row * 10 + column] = left(row, column)
                inputValues[80 + column * 8 + row] = right(row, column)
                inputValues[144 + row * 8 + column] = accum(row, column)
                let product = (0..<8).reduce(accum(row, column)) { $0 + left(row, $1) * right($1, column) }
                expected[row * 10 + column] = product
                expected[80 + row * 8 + column] = accum(row, column)
            }
        }
        let input = try await device.createBuffer(.init(size: 832, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 576, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 576, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: inputValues.map(\.bitPattern)),
            .write(buffer: output, offset: 0, words: Array(repeating: Float(-42).bitPattern, count: expected.count)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 576),
            .read(buffer: read, byteCount: 576),
        ])
        let result = try await ticket.wait()
        #expect(result == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
