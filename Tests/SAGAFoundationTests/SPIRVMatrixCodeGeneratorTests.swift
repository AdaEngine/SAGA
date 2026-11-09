import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvMatrixInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f, 6>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 10>;

    @compute @workgroup_size(1) fn run() {
        let a = mat2x3f(inputs[0].xyz, inputs[1].xyz);
        let b = mat4x2f(inputs[2].xy, inputs[3].xy, inputs[4].xy, inputs[5].xy);
        let product = a * b;
        let column = a * vec2f(2.0, 3.0);
        let row = vec3f(2.0, 3.0, 4.0) * a;
        let transposed = transpose(a);
        let scaled = (a * 2.0) + (3.0 * a) - a;
        output[0] = vec4f(product[0], -1.0);
        output[1] = vec4f(product[1], -1.0);
        output[2] = vec4f(product[2], -1.0);
        output[3] = vec4f(product[3], -1.0);
        output[4] = vec4f(column, -1.0);
        output[5] = vec4f(row, 0.0, -1.0);
        output[6] = vec4f(transposed[0], transposed[1]);
        output[7] = vec4f(transposed[2], 0.0, -1.0);
        output[8] = vec4f(scaled[0], -1.0);
        output[9] = vec4f(scaled[1], -1.0);
    }
    """)

private func exportMatrix(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvMatrixInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvMatrixCodeGeneratorRoundTripsRectangularMatrixOperations(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportMatrix(compiler)
    try #require(words.count > 5)
    var opcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        opcodes.insert(words[offset] & 0xffff)
        offset += count
    }
    // OpTypeMatrix, OpTranspose, MatrixTimesScalar, VectorTimesMatrix, MatrixTimesVector, MatrixTimesMatrix.
    #expect(opcodes.isSuperset(of: [24, 84, 143, 144, 145, 146]))
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [96, 160])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

#if canImport(Metal)
@Test func spirvMatrixCodeGeneratorPreservesColumnMajorResultsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportMatrix(compiler)), to: .msl)
    let inputs: [Float] = [1, 2, 3, -99, 4, 5, 6, -99, 7, 8, -99, -99,
                           9, 10, -99, -99, 11, 12, -99, -99, 13, 14, -99, -99]
    let expected: [Float] = [39, 54, 69, -1, 49, 68, 87, -1, 59, 82, 105, -1, 69, 96, 123, -1,
                             14, 19, 24, -1, 20, 47, 0, -1, 1, 4, 2, 5, 3, 6, 0, -1,
                             4, 8, 12, -1, 16, 20, 24, -1]
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 96, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 160, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 160, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: inputs.map(\.bitPattern)),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 160), .read(buffer: readback, byteCount: 160),
        ]).wait()
        #expect(snapshots == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func spirvMatrixCodeGeneratorPreservesStorageColumnPaddingOnMetal() async throws {
    let source = ShaderInput.wgsl("""
        struct Packet { matrices: array<mat2x3f, 2>, sentinel: u32 }
        @group(0) @binding(0) var<storage, read_write> data: Packet;
        @compute @workgroup_size(1) fn run() {
            let a = data.matrices[0];
            let b = data.matrices[1];
            data.matrices[1] = a * 2.0 + b;
        }
        """)
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    try #require(words.count > 5)
    var columnMajor = false
    var matrixStride = false
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        if words[offset] & 0xffff == 72 { // OpMemberDecorate
            if count == 4 && words[offset + 3] == 5 { columnMajor = true }
            if count == 5 && words[offset + 3] == 7 && words[offset + 4] == 16 { matrixStride = true }
        }
        offset += count
    }
    #expect(columnMajor && matrixStride)
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [80])
    func column(_ x: Float, _ y: Float, _ z: Float, _ padding: UInt32) -> [UInt32] {
        [x.bitPattern, y.bitPattern, z.bitPattern, padding]
    }
    let first = column(1, 2, 3, 701) + column(4, 5, 6, 702)
    let sentinel: [UInt32] = [0xdeadbeef, 900, 901, 902]
    let initial = first + column(7, 8, 9, 703) + column(10, 11, 12, 704) + sentinel
    let expected = first + column(9, 12, 15, 703) + column(18, 21, 24, 704) + sentinel
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let data = try await device.createBuffer(.init(size: 80, usage: [.storage, .copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: data, offset: 0, words: initial),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data], workgroups: .init(x: 1)),
            .copy(source: data, destination: readback, byteCount: 80), .read(buffer: readback, byteCount: 80),
        ]).wait()
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
