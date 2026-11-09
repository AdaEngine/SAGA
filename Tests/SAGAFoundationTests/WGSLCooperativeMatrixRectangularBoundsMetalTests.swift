import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

#if canImport(Metal)
@Test func wgslRectangularCooperativeOutputAndTruncatedBuffersOnMetal() async throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let a = coopLoadT<coop_mat8x16<f32, A>>(&input[0]);
            let b = coopLoadT<coop_mat16x8<f32, B>>(&input[128]);
            let c = coopLoadT<coop_mat16x16<f32, C>>(&input[256]);
            let product = coopMultiplyAdd(a, b, c);
            coopStoreT(product, &output[0]);
            let truncated = coopLoadT<coop_mat8x16<f32, A>>(&input[500]);
            coopStoreT(coopMultiplyAdd(truncated, b, c), &output[256]);
            coopStoreT(product, &output[512]);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    func left(_ row: Int, _ column: Int) -> Float { Float((row + 2 * column) % 5 - 2) }
    func right(_ row: Int, _ column: Int) -> Float { Float((2 * row + 3 * column) % 7 - 3) }
    var values = Array(repeating: Float(0), count: 512)
    var expected = Array(repeating: Float(-42), count: 513)
    for row in 0..<16 {
        for column in 0..<8 { values[row * 8 + column] = left(row, column) }
    }
    for row in 0..<8 {
        for column in 0..<16 { values[128 + row * 16 + column] = right(row, column) }
    }
    for row in 0..<16 {
        for column in 0..<16 {
            let c = Float(row - column)
            values[256 + row * 16 + column] = c
            expected[row * 16 + column] = (0..<8).reduce(c) { $0 + left(row, $1) * right($1, column) }
            expected[256 + row * 16 + column] = c
        }
    }
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createBuffer(.init(size: 2048, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 2052, usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 2052, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: source, offset: 0, words: values.map(\.bitPattern)),
            .write(buffer: output, offset: 0, words: Array(repeating: Float(-42).bitPattern, count: 513)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): source, .init(binding: 1): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 2052),
            .read(buffer: read, byteCount: 2052),
        ])
        #expect(try await ticket.wait() == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
