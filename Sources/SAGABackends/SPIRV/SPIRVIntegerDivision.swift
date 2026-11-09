import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func writeIntegerQuotient(
        _ lhs: UInt32, _ rhs: UInt32, shape: IRBitcastShape, remainder: Bool
    ) throws -> UInt32 {
        let width = shape.scalar.width
        let isUnsigned = shape.scalar.kind == .uint
        let unsigned = try scalarOrVectorTypeID(.init(kind: .uint, width: width), size: shape.size)
        let opcode: UInt32 = remainder ? 137 : 134 // OpUMod / OpUDiv
        if isUnsigned, let bits = constantWords[rhs], bits != 0 {
            return try rawValueInstruction(opcode, type: unsigned, operands: [lhs, rhs])
        }

        let boolean = try scalarOrVectorTypeID(.bool, size: shape.size)
        let zero = try zeroID(typeID: unsigned)
        let unit: IRLiteral = switch width {
        case 2: .u16(1)
        case 4: .u32(1)
        case 8: .u64(1)
        default: throw ShaderFailure.invalidHandle
        }
        let one = try splatLiteralID(unit, type: unsigned, size: shape.size)
        if isUnsigned {
            let divisor = try nonzeroDivisor(rhs, zero: zero, one: one, unsigned: unsigned, boolean: boolean)
            return try rawValueInstruction(opcode, type: unsigned, operands: [lhs, divisor])
        }

        // Unsigned magnitudes also define INT_MIN / -1 and avoid negative OpSRem operands on Vulkan 1.1.
        let signed = try scalarOrVectorTypeID(.init(kind: .sint, width: width), size: shape.size)
        let signedZero = try zeroID(typeID: signed)
        let left = try signedMagnitude(lhs, signedZero: signedZero, zero: zero, unsigned: unsigned, boolean: boolean)
        let right = try signedMagnitude(rhs, signedZero: signedZero, zero: zero, unsigned: unsigned, boolean: boolean)
        let divisor = try nonzeroDivisor(right.magnitude, zero: zero, one: one, unsigned: unsigned, boolean: boolean)
        let bits = try rawValueInstruction(opcode, type: unsigned, operands: [left.magnitude, divisor])
        let negative = try remainder ? left.negative : rawValueInstruction(165, type: boolean, operands: [left.negative, right.negative])
        let negated = try rawValueInstruction(130, type: unsigned, operands: [zero, bits])
        let result = try rawValueInstruction(169, type: unsigned, operands: [negative, negated, bits])
        return try rawValueInstruction(124, type: signed, operands: [result]) // OpBitcast
    }

    private mutating func nonzeroDivisor(
        _ divisor: UInt32, zero: UInt32, one: UInt32, unsigned: UInt32, boolean: UInt32
    ) throws -> UInt32 {
        let isZero = try rawValueInstruction(170, type: boolean, operands: [divisor, zero])
        return try rawValueInstruction(169, type: unsigned, operands: [isZero, one, divisor])
    }

    private mutating func signedMagnitude(
        _ value: UInt32, signedZero: UInt32, zero: UInt32, unsigned: UInt32, boolean: UInt32
    ) throws -> (magnitude: UInt32, negative: UInt32) {
        let negative = try rawValueInstruction(177, type: boolean, operands: [value, signedZero])
        let bits = try rawValueInstruction(124, type: unsigned, operands: [value])
        let negated = try rawValueInstruction(130, type: unsigned, operands: [zero, bits])
        let magnitude = try rawValueInstruction(169, type: unsigned, operands: [negative, negated, bits])
        return (magnitude, negative)
    }
}
