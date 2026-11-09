#if canImport(Metal)
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

@Test(arguments: 0..<32)
func wgslRectangularHalfCooperativeMatricesExecuteOnMetal(_ mask: Int) async throws {
    let halfAccumulator = mask & 1 != 0
    let transposeA = mask & 2 != 0
    let transposeB = mask & 4 != 0
    let transposeC = mask & 8 != 0
    let transposeOutput = mask & 16 != 0
    let scalar = halfAccumulator ? "f16" : "f32"
    func load(_ transpose: Bool) -> String { transpose ? "coopLoadT" : "coopLoad" }
    let store = transposeOutput ? "coopStoreT" : "coopStore"
    let input = ShaderInput.wgsl("""
        enable f16, wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> input: array<f16>;
        @group(0) @binding(1) var<storage, read> initial: array<\(scalar)>;
        @group(0) @binding(2) var<storage, read_write> output: array<\(scalar)>;
        @compute @workgroup_size(32) fn main() {
            let a = \(load(transposeA))<coop_mat16x8<f16, A>>(&input[0], \(transposeA ? 19 : 11)u);
            let b = \(load(transposeB))<coop_mat8x16<f16, B>>(&input[200], \(transposeB ? 11 : 19)u);
            let c = \(load(transposeC))<coop_mat8x8<\(scalar), C>>(&initial[0], 11u);
            let product = coopMultiplyAdd(a, b, c);
            \(store)(product, &output[0], 11u);
            let truncated = \(load(transposeA))<coop_mat16x8<f16, A>>(&input[398], \(transposeA ? 19 : 11)u);
            \(store)(coopMultiplyAdd(truncated, b, c), &output[88], 11u);
            \(store)(product, &output[176], 11u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    func left(_ r: Int, _ c: Int) -> Float { Float((r + 2 * c) % 5 - 2) }
    func right(_ r: Int, _ c: Int) -> Float { Float((2 * r + 3 * c) % 7 - 3) }
    func index(_ r: Int, _ c: Int, _ stride: Int, _ transpose: Bool) -> Int {
        transpose ? r * stride + c : c * stride + r
    }
    var values = Array(repeating: Float16(-99), count: 400)
    var initial = Array(repeating: Float(-77), count: 88)
    var expected = Array(repeating: Float(-42), count: 178)
    for row in 0..<8 {
        for column in 0..<16 { values[index(row, column, transposeA ? 19 : 11, transposeA)] = Float16(left(row, column)) }
    }
    for row in 0..<16 {
        for column in 0..<8 { values[200 + index(row, column, transposeB ? 11 : 19, transposeB)] = Float16(right(row, column)) }
    }
    for row in 0..<8 {
        for column in 0..<8 {
            let c = Float(row - column)
            initial[index(row, column, 11, transposeC)] = c
            expected[88 + index(row, column, 11, transposeOutput)] = c
            expected[index(row, column, 11, transposeOutput)] = (0..<16).reduce(c) { $0 + left(row, $1) * right($1, column) }
        }
    }
    func packHalf(_ values: [Float16]) -> [UInt32] {
        stride(from: 0, to: values.count, by: 2).map { UInt32(values[$0].bitPattern) | UInt32(values[$0 + 1].bitPattern) << 16 }
    }
    let sourceWords = packHalf(values)
    let initialWords = halfAccumulator ? packHalf(initial.map(Float16.init)) : initial.map(\.bitPattern)
    let outputWords = halfAccumulator ? packHalf(Array(repeating: Float16(-42), count: 178)) : Array(repeating: Float(-42).bitPattern, count: 178)
    let expectedWords = halfAccumulator ? packHalf(expected.map(Float16.init)) : expected.map(\.bitPattern)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createBuffer(.init(size: UInt64(sourceWords.count * 4), usage: [.storage, .copyDestination]))
        let initialBuffer = try await device.createBuffer(.init(size: UInt64(initialWords.count * 4), usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: UInt64(outputWords.count * 4), usage: [.storage, .copySource, .copyDestination]))
        let read = try await device.createBuffer(.init(size: UInt64(outputWords.count * 4), usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: source, offset: 0, words: sourceWords),
            .write(buffer: initialBuffer, offset: 0, words: initialWords),
            .write(buffer: output, offset: 0, words: outputWords),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): source, .init(binding: 1): initialBuffer, .init(binding: 2): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: UInt64(outputWords.count * 4)),
            .read(buffer: read, byteCount: UInt64(outputWords.count * 4)),
        ])
        #expect(try await ticket.wait() == [expectedWords])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
