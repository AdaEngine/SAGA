import SAGACompiler
import SAGAShaderIR
import Testing

@Test func wgslRectangularCooperativeMatricesEmitMetalTiles() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoadT<coop_mat16x8<f32, A>>(&input[0], 19u);
            let b = coopLoad<coop_mat8x16<f32, B>>(&input[152], 18u);
            let c = coopLoadT<coop_mat8x8<f32, C>>(&input[296]);
            coopStoreT(coopMultiplyAdd(a, b, c), &output[0]);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    guard case .text(.msl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.split(separator: "simdgroup_multiply_accumulate(").count == 3)
    #expect(source.contains("+ 16ul <= ulong("))
}

#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes

@Test func wgslRectangularCooperativeMatricesExecuteOnMetal() async throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoadT<coop_mat16x8<f32, A>>(&input[0], 19u);
            let b = coopLoad<coop_mat8x16<f32, B>>(&input[152], 18u);
            let c = coopLoadT<coop_mat8x8<f32, C>>(&input[296]);
            coopStoreT(coopMultiplyAdd(a, b, c), &output[0], 10u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    var values = Array(repeating: Float(-99), count: 360)
    var expected = Array(repeating: Float(-42), count: 80)
    func left(_ r: Int, _ c: Int) -> Float { Float((r + 2 * c) % 5 - 2) }
    func right(_ r: Int, _ c: Int) -> Float { Float((2 * r + 3 * c) % 7 - 3) }
    for row in 0..<8 {
        for column in 0..<16 { values[row * 19 + column] = left(row, column) }
    }
    for row in 0..<16 {
        for column in 0..<8 { values[152 + column * 18 + row] = right(row, column) }
    }
    for row in 0..<8 {
        for column in 0..<8 {
            values[296 + row * 8 + column] = Float(row - column)
            expected[row * 10 + column] = (0..<16).reduce(Float(row - column)) { $0 + left(row, $1) * right($1, column) }
        }
    }
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createBuffer(.init(size: 1440, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 320, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 320, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: source, offset: 0, words: values.map(\.bitPattern)),
            .write(buffer: output, offset: 0, words: Array(repeating: Float(-42).bitPattern, count: 80)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): source, .init(binding: 1): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 320),
            .read(buffer: read, byteCount: 320),
        ])
        #expect(try await ticket.wait() == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
