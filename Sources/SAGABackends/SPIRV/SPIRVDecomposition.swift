import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func writeDecomposition(
        _ builtin: IRMathFunction, operand: UInt32, type: IRHandle<IRType>
    ) throws -> UInt32 {
        let instruction: UInt32
        switch builtin {
        case .modf: instruction = 36 // ModfStruct
        case .frexp: instruction = 52 // FrexpStruct
        default: throw ShaderFailure.invalidHandle
        }
        return try glslMathInstruction(instruction, type: typeID(type), operands: [operand])
    }
}
