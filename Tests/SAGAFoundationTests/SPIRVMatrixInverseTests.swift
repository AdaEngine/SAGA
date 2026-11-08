import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

extension SPIRVFrontendTests {
    static func inverseMatrixModule(scale: Float = 1) -> [UInt32] {
        var words = head
        words[3] = 1024
        words += i(17, 1) + i(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0) + i(14, 0, 1)
        words += i(15, 5, 15, 0x6e69616d, 0) + i(16, 15, 17, 1, 1, 1)
        words += i(71, 10, 2) + i(71, 14, 34, 0) + i(71, 14, 33, 0)
        for index: UInt32 in 0..<29 { words += i(72, 10, index, 35, index * 4) }
        func op(_ opcode: UInt32, _ operands: [UInt32]) -> [UInt32] {
            [UInt32(operands.count + 1) << 16 | opcode] + operands
        }
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 2), i(23, 5, 3, 3), i(23, 6, 3, 4),
            i(24, 7, 4, 2), i(24, 8, 5, 3), i(24, 9, 6, 4),
            op(30, [10] + Array(repeating: 3, count: 29)),
            i(32, 11, 12, 10), i(32, 12, 12, 3), i(33, 13, 1),
        ] {
            words += instruction
        }
        let matrices: [[[Float]]] = [
            [[2, 0], [1, 4]],
            [[1, 0, 5], [2, 1, 6], [3, 4, 0]],
            [[2, 0, 0, 0], [1, 4, 0, 0], [0, 2, 8, 0], [0, 0, 4, 16]],
        ]
        var next: UInt32 = 100
        var columns: [[UInt32]] = []
        for (matrixIndex, matrix) in matrices.enumerated() {
            var columnIDs: [UInt32] = []
            for column in matrix {
                var elements: [UInt32] = []
                for value in column {
                    words += i(43, 3, next, (value * scale).bitPattern)
                    elements.append(next)
                    next += 1
                }
                words += op(44, [UInt32(4 + matrixIndex), next] + elements)
                columnIDs.append(next)
                next += 1
            }
            columns.append(columnIDs)
        }
        let smallMatrix = next
        next += 1
        words += op(44, [7, smallMatrix] + columns[0])
        let indexBase = next
        for index: UInt32 in 0..<29 { words += i(43, 2, indexBase + index, index) }
        next += 29
        words += i(59, 11, 14, 12) + i(54, 1, 15, 0, 13) + i(248, 16)
        var outputIndex: UInt32 = 0
        for (index, columnIDs) in columns.enumerated() {
            let type = UInt32(7 + index)
            let matrix: UInt32
            if index == 0 {
                matrix = smallMatrix
            } else {
                matrix = next
                next += 1
                words += op(80, [type, matrix] + columnIDs)
            }
            let inverted = next
            next += 1
            words += i(12, type, inverted, 80, 34, matrix)
            let size = UInt32(columnIDs.count)
            for column: UInt32 in 0..<size {
                for row: UInt32 in 0..<size {
                    words += i(81, 3, next, inverted, column, row)
                    words += i(65, 12, next + 1, 14, indexBase + outputIndex)
                    words += i(62, next + 1, next)
                    outputIndex += 1
                    next += 2
                }
            }
        }
        return words + i(253) + i(56)
    }

    @Test func spirvMatrixInverseRejectsRectangularOperands() throws {
        var words = Self.head + Self.i(17, 1) + Self.i(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0) + Self.i(14, 0, 1)
        for instruction in [
            Self.i(19, 1), Self.i(22, 2, 32), Self.i(23, 3, 2, 3), Self.i(24, 4, 3, 2), Self.i(33, 5, 1, 4),
            Self.i(54, 1, 10, 0, 5), Self.i(55, 4, 11), Self.i(248, 12),
            Self.i(12, 4, 13, 80, 34, 11), Self.i(253), Self.i(56),
        ] {
            words += instruction
        }
        try fails(words)
    }
}

private let inverseMatrixExpected: [Float] = [
    0.5, 0, -0.125, 0.25,
    -24, 20, -5, 18, -15, 4, 5, -4, 1,
    0.5, 0, 0, 0, -0.125, 0.25, 0, 0,
    0.03125, -0.0625, 0.125, 0, -0.0078125, 0.015625, -0.03125, 0.0625,
]

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvMatrixInverseTranslatesSquareMatrices(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(SPIRVFrontendTests.inverseMatrixModule()), to: format)
    guard case .text(let actual, let text) = shader.artifact else { Issue.record("Expected shader text"); return }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(text)) }
}

@Test func spirvMatrixInverseDoesNotAddAWGSLBuiltin() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl("@compute @workgroup_size(1) fn run() { _ = inverse(mat2x2f()); }"))
    }
    _ = try compiler.parseAndValidate(.wgsl("""
        fn inverse(value: f32) -> f32 { return 1.0 / value; }
        @compute @workgroup_size(1) fn run() { _ = inverse(2.0); }
        """))
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func spirvMatrixInverseWidensHalfIntermediates(_ format: ShaderOutputFormat) throws {
    var words = SPIRVFrontendTests.head
    func op(_ opcode: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | opcode] + operands
    }
    for instruction in [
        op(17, 1), op(17, 9), op(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0), op(14, 0, 1),
        op(15, 5, 20, 0x6e69616d, 0), op(16, 20, 17, 1, 1, 1),
        op(19, 1), op(22, 2, 16), op(23, 3, 2, 2), op(24, 4, 3, 2), op(33, 5, 1),
        op(43, 2, 6, 0x3c00), op(46, 2, 7), op(44, 3, 8, 6, 7), op(44, 3, 9, 6, 6),
        op(44, 4, 10, 8, 9), op(54, 1, 20, 0, 5), op(248, 21), op(12, 4, 22, 80, 34, 10), op(253), op(56),
    ] {
        words += instruction
    }
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(words), to: format)
    #expect(shader.requiresShaderF16)
    guard case .text(_, let code) = shader.artifact else { Issue.record("Expected shader text"); return }
    if format == .wgsl {
        #expect(code.contains("vec2<f32>") && code.contains("mat2x2<f16>"))
        _ = try compiler.parseAndValidate(.wgsl(code), capabilities: .referenceDefault.union(.shaderFloat16))
    }
}

#if canImport(Metal)
@Test(arguments: [Float(1), 1e-12, 1e12])
func spirvMatrixInversePreservesColumnsAndScaleOnMetal(_ scale: Float) async throws {
    let input = ShaderInput.spirv(SPIRVFrontendTests.inverseMatrixModule(scale: scale))
    let shader = try IndependentShaderCompiler().compileCompute(input, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 116, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 116, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 116),
            .read(buffer: readback, byteCount: 116),
        ]).wait()
        let words = try #require(result.first)
        try #require(words.count == inverseMatrixExpected.count)
        for (word, expected) in zip(words, inverseMatrixExpected) {
            #expect(abs(Float(bitPattern: word) * scale - expected) <= max(1, abs(expected)) * 0.0008)
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
